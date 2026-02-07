
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

    K(h, params) = K_iso(warp(params[warp_param_indices], h), 
                         params[sdf_param_indices]),

and

    K_iso(t) = ∫_{ℜ} cos(2 π ω t) cfg.sdf(ω, params[sdf_param_indices]) d ω.

The recommended interface to this object is to write your spectral density in scalar form as

    sdf(ω, param1, ..., paramk) = [...] # your cool model
    warp(warp_params, h)        = [...] # your warping model if necessary
    psdf = ParametricFunction(sdf, (param1, ..., paramk)) # note the tuple!
    global_params = vcat(warp_params, sdf_params)
    cfg = AdaptiveKernelConfig(psdf; tol=1e-12) # also other args available
    lags = [...] # the lags at which you will want to evaluate your kernel K
    model = SpectralModel(cfg, warp, tuple((1:length(warp_params))...), 
                         tuple((length(warp_params)+1):length(sdf_params)...), 
                         0, lags)

"""
struct SpectralModel{S,dS,W,P1,P2,L}
  cfg::AdaptiveKernelConfig{S,dS}
  warp::W
  warp_param_indices::NTuple{P1,Int64}
  sdf_param_indices::NTuple{P2,Int64}
  singularity_param_index::Int64
  lags::Vector{L}
end

# TODO (cg 2026/02/06 18:08): test all of this below.

struct SpectralKernel{L,T}
  store::Dict{L,T}
end

function gen_kernel(sm::SpectralModel, params)
  sdf_params  = params[sm.sdf_param_indices]
  warp_params = params[sm.sdf_param_indices]
  new_kernel  = ParametricFunction(sm.cfg.f.fn, tuple(sdf_params...))
  new_cfg     = gen_new_sdf_config(sm.cfg, new_kernel)
  warp_lags   = [sm.warp(warp_params, lagj) for lagj in sm.lags]
  values      = kernel_values(new_cfg, warp_lags; verbose=false)[1]
  SpectralKernel(Dict(sm.lags, values))
end

function (sk::SpectralKernel{L,T})(x::L, y::L) where{L,T}
  lag = x-y
  res = get(sk.store, lag,  nothing)
  !isnothing(res) && return res
  res = get(sk.store, -lag, nothing)
  !isnothing(res) && return res
  error("Lag $(lag) not in the `SpectralKernel` lookup table.")
end

