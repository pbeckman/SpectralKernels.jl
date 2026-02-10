
# TODO (cg 2026/02/06 18:02): maybe a compile-time check to make sure that the
# eltype of lags (like, if it is a Vector{SVector{2,Float64}}) matches the
# reported dimension in the AdaptiveKernelConfig object.
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
  pts_pairs::Vector{Tuple{Int64, Int64}} # defaults to all unique pairs.
end

struct SpectralKernel{L,T}
  store::Dict{Tuple{L,L},T}
end

function gen_new_sdf_config(sm::SpectralModel, params)
end

function gen_kernel_setup(sm::SpectralModel, params)
  # set up the new integration cfg.
  sdf_params  = [params[j] for j in sm.sdf_param_indices] 
  new_sdf     = ParametricFunction(sm.cfg.f, tuple(sdf_params...))
  new_cfg     = gen_new_sdf_config(sm.cfg, new_sdf)
  # compute the warped points.
  warp_params = [params[j] for j in sm.warp_param_indices]
  warp_pts    = [sm.warp(warp_params, ptj) for ptj in sm.pts]
  # using the sm.pts_pairs (particularly relevant if you only need O(n) elements
  # of the covariance matrix Σ), create the pairs of both raw and warped points.
  # Since kernel_values wants the lags to be sorted monotonically, we also
  # generate the sortperm for warp_lags and permute both the raw_pairs (used for
  # indexing the kernel) and warp_lags (used in kernel_values).
  raw_pairs   = [(sm.pts[jk[1]],   sm.pts[jk[2]])        for jk in sm.pts_pairs]
  warp_lags   = [norm(warp_pts[jk[1]] - warp_pts[jk[2]]) for jk in sm.pts_pairs]
  # apply the permutation so that the warp_lags are sorted, as required by
  # kernel_values.
  sp          = sortperm(warp_lags)
  warp_lags   = warp_lags[sp]
  raw_pairs   = raw_pairs[sp]
  # return the necessary internal objects.
  (new_cfg, warp_lags, raw_pairs, warp_params, sp)
end

# TODO (cg 2026/02/09 13:28): this kernel should maybe have a stationary=true
# argument to it or something. For points on a lattice, for example, a lot of
# redundant compuations will happen in its current form.
#
# TODO (cg 2026/02/09 13:45): this is the one function that I will write a
# custom rule for.
function gen_kernel(sm::SpectralModel, params)
  (new_cfg, warp_lags, raw_pairs, _, _) = gen_kernel_setup(sm, params)
  values = kernel_values(new_cfg, warp_lags; verbose=false)[1]
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

