
"""
`warping_gradients(warping_function::W, xs, warping_params; backend)`

An **internal** routine for evaluating the derivative of a kernel specified by
the spectral density in `cfg` with respect to parameters that can be isolated to
argument transformations. In particular, if your kernel is 

K_θ(h) = K_θ1( ||w_θ2(h)|| )

where θ = θ_1 ∪ θ_2, then you can use the chain rule to differentiate 

(d / d θ2) K_θ(h) = K'_θ1( ||w_θ2(h)|| ) ⋅ (d / dθ2 ||w_θ2(h)||).

Using this identity reduces the number of times expensive integrals need to be
re-computed. 

The warping function is presumed to have the signature

`warping_norm(θ2::Vector{T}, h)::T`,

as `DifferentiationInterface.jl` (I gather) requires constant values to be in
the second or later position.
"""
function warping_gradients(warping_norm::W, xs, warping_params; backend,
                           multipliers=ones(length(xs))) where{W}
  prep  = prepare_gradient(warping_norm, backend, warping_params, 
                           Constant(one(eltype(xs))))
  grads = [Vector{Float64}(undef, length(warping_params)) for _ in eachindex(xs)]
  foreach(zip(grads, xs, multipliers)) do (g, r, c)
    gradient!(warping_norm, g, prep, backend, warping_params, Constant(r))
    g .*= c
  end
  grads
end

# TODO (cg 2026/02/06 15:11): should this just take warped_xs? I think no, but
# something to think about.
function kernel_warping_gradients(cfg::AdaptiveKernelConfig{S,dS},
                                  warping_norm::W, xs, 
                                  warping_params; backend) where{S,dS,W}
  dcfg    = gen_derivative_config(cfg)
  dvalues = kernel_values(dcfg, warping_norm.(Ref(warping_params), xs))[1]
  warping_gradients(warping_norm, xs, warping_params; backend,
                    multipliers = dvalues)
end

# TODO (cg 2026/02/06 17:25): this does _not_ work for the singularity
# parameter, which needs to be handled separately.
function kernel_sdf_derivatives(cfg::AdaptiveKernelConfig{ParametricFunction{S,P},dS},
                                xs; backend) where{S,P,dS}
  # get the derivatives of the parametric sdf.
  psdf   = cfg.f
  dsdfs  = derivatives(psdf, backend)
  fnames = fieldnames(AdaptiveKernelConfig)
  repeated_fields = getfield.(Ref(cfg), fnames[3:end])
  # now for each one, make a new configuration and do the integration.
  derivs = map(dsdfs) do dsdf_dj
    cfgj = AdaptiveKernelConfig(dsdf_dj, nothing, repeated_fields...)
    kernel_values(cfgj, xs; verbose=false)[1]
  end
end

