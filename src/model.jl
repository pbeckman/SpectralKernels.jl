
struct NoWarping end

(nw::NoWarping)(params, x) = x

"""
    SpectralModel(cfg::AdaptiveKernelConfig,
                  warp,
                  warp_param_indices::NTuple{P1,Int64},
                  sdf_param_indices::NTuple{P2,Int64},
                  singularity_param_index::Int64,
                  lags)

An object specifying a model with a covariance function given by

    K(x, y, params) = K_iso(|| 
                               warp(params[warp_param_indices], x)
                               -
                               warp(params[warp_param_indices], y)
                            ||,
                            params[sdf_param_indices]),

and

    K_iso(t) = ∫_{ℜ} cos(2 π ω t) cfg.sdf(ω, params[sdf_param_indices]) d ω.

The recommended interface to this object is to write your spectral density in scalar form as

    sdf(ω, param1, ..., paramk) = [...] # your cool model
    warp(warp_params, h)        = [...] # your warping model (if necessary)
    psdf = ParametricFunction(sdf, (param1, ..., paramk)) # note the tuple!
    global_params = vcat(warp_params, sdf_params)
    cfg = AdaptiveKernelConfig(psdf; tol=1e-12) # also other args available
    lags = [...] # the lags at which you will want to evaluate your kernel K
    model = SpectralModel(cfg, warp, tuple((1:length(warp_params))...), 
                         tuple((length(warp_params)+1):length(sdf_params)...), 
                         0, lags)

"""
Base.@kwdef struct SpectralModel{S,dS,W,P1,P2,L}
  cfg::AdaptiveKernelConfig{S,dS}
  warp::W
  sdf_param_indices::NTuple{P2,Int64}
  warp_param_indices::NTuple{P1,Int64}
  singularity_param_index::Int64
  pts::Vector{L}
  kernel_index_pairs::Vector{Tuple{Int64, Int64}} # defaults to all unique pairs.
  verbose::Bool
end

function dense_index_pairs(pts)
  allpairs = vec(collect(Iterators.product(eachindex(pts), eachindex(pts)))) 
  filter!(x->x[1] <= x[2], allpairs)
  allpairs
end

# TODO (cg 2026/02/10 14:38): At present undocumented while we discuss what this
# front-end constructor should look like.
function SpectralModel(sdf, pts; warp=NoWarping(),
                       kernel_index_pairs=dense_index_pairs(pts),
                       sdf_param_indices, warp_param_indices=(),
                       singularity_param_index=0, verbose=false, kwargs...)
  cfg = AdaptiveKernelConfig(sdf; dim=length(first(pts)), kwargs...)
  SpectralModel(cfg, warp, tuple(sdf_param_indices...),
                tuple(warp_param_indices...), singularity_param_index,
                pts, kernel_index_pairs, verbose)
end


struct SpectralKernel{L,T}
  store::Dict{Tuple{L,L},T}
end

function gen_kernel_setup(sm::SpectralModel, params)
  # set up the new integration cfg.
  sdf_params  = [params[j] for j in sm.sdf_param_indices] 
  new_sdf     = ParametricFunction(sm.cfg.f, tuple(sdf_params...))
  new_cfg     = gen_new_sdf_config(sm.cfg, new_sdf)
  # compute the warped points.
  warp_params = [params[j] for j in sm.warp_param_indices]
  warp_pts    = [sm.warp(warp_params, ptj) for ptj in sm.pts]
  # using the sm.kernel_index_pairs (particularly relevant if you only need O(n) elements
  # of the covariance matrix Σ), create the pairs of both raw and warped points.
  raw_pairs   = [(sm.pts[jk[1]],   sm.pts[jk[2]])        for jk in sm.kernel_index_pairs]
  warp_lags   = [norm(warp_pts[jk[1]] - warp_pts[jk[2]]) for jk in sm.kernel_index_pairs]
  # return the necessary internal objects.
  (new_cfg, warp_lags, raw_pairs, warp_params)
end

# TODO (cg 2026/02/09 13:28): this kernel should maybe have a stationary=true
# argument to it or something. For points on a lattice, for example, a lot of
# redundant computations will happen in its current form.
#
# TODO (cg 2026/02/09 13:45): this is the one function that I will write a
# custom rule for.
function gen_kernel(sm::SpectralModel, params)
  (new_cfg, warp_lags, raw_pairs, _) = gen_kernel_setup(sm, params)
  values = kernel_values(new_cfg, warp_lags; verbose=sm.verbose)[1]
  SpectralKernel(Dict(zip(raw_pairs, values)))
end

function (sk::SpectralKernel{L,T})(x::L, y::L) where{L,T}
  res = get(sk.store, (x, y),  nothing)
  !isnothing(res) && return res
  res = get(sk.store, (y, x), nothing)
  !isnothing(res) && return res
  error("Point pair ($(x), $(y)) not in the `SpectralKernel` lookup table.")
end

# just swallow the potential third argument, which may be parameters or
# something. Maybe SpectralKernel should keep the parameters and allow a safety
# check, but for now since it is just internal use I'll keep things fast and wild.
(sk::SpectralKernel{L,T})(x::L, y::L, arg3) where{L,T} = sk(x, y)

