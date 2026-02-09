
struct WarpingDiffNorm{W}
  warp::W
end

function (wdn::WarpingDiffNorm{W})(params, x, y) where{W}
  norm(wdn.warp(params, x) - wdn.warp(params, y))
end

"""
`warping_gradients(warping_function::W, xs, warping_params; backend)`

An **internal** routine for evaluating the derivative of a kernel specified by
the spectral density in `cfg` with respect to parameters that can be isolated to
argument transformations. In particular, if your kernel is 

K_θ(x - y) = K_θ1( || w_θ2(x) - w_θ2(y) || )

where θ = θ_1 ∪ θ_2, then you can use the chain rule to differentiate 

(d / d θ2) K_θ(h) = K'_θ1( || w_θ2(x) - w_θ2(y) || ) ⋅ (d / dθ2 || w_θ2(x) - w_θ2(y) ||).

Using this identity reduces the number of times expensive integrals need to be
re-computed. 

The warping function is presumed to have the signature

`warping_norm(θ2::Vector{T}, h)::T`,

as `DifferentiationInterface.jl` (I gather) requires constant values to be in
the second or later position.
"""
function warping_gradients(warp::W, raw_xs_pairs, warping_params; backend,
                           multipliers=ones(length(raw_xs_pairs))) where{W}
  warping_norm = WarpingDiffNorm(warp)
  prep  = prepare_gradient(warping_norm, backend, warping_params, 
                           Constant(raw_xs_pairs[1][1]), Constant(raw_xs_pairs[1][2]))
  grads = [Vector{Float64}(undef, length(warping_params)) for _ in eachindex(raw_xs_pairs)]
  foreach(zip(grads, raw_xs_pairs, multipliers)) do (g, xy, c)
    (x, y) = (xy[1], xy[2])
    gradient!(warping_norm, g, prep, backend, warping_params, Constant(x), Constant(y))
    g .*= c
  end
  grads
end

# IMPORTANT: warped_lags and raw_xs_pairs must be in the same order, in the
# sense that if warped_lags gets sorted to be a valid input to kernel_values,
# rax_xs_pairs must ALSO be sorted. But based on how they come in here, it is
# not possible for this code to enforce that joint permutation.
function kernel_warping_gradients(cfg::AdaptiveKernelConfig{S,dS},
                                  warp::W, raw_xs_pairs,
                                  warped_lags, warping_params; 
                                  backend) where{S,dS,W}
  dcfg    = gen_derivative_config(cfg)
  dvalues = kernel_values(dcfg, warped_lags)[1]
  warping_gradients(warp, raw_xs_pairs, warping_params; backend,
                    multipliers = dvalues)
end

# TODO (cg 2026/02/06 17:25): this does _not_ work for the singularity
# parameter, which needs to be handled separately.
function kernel_sdf_derivatives(cfg::AdaptiveKernelConfig{ParametricFunction{S,P},dS},
                                xs; backend) where{S,P,dS}
  # get the derivatives of the parametric sdf.
  dsdfs  = derivatives(cfg.f, backend)
  # now for each one, make a new configuration and do the integration.
  derivs = map(dsdfs) do dsdf_dj
    cfgj = gen_new_sdf_config(cfg, dsdf_dj)
    kernel_values(cfgj, xs; verbose=false)[1]
  end
end

