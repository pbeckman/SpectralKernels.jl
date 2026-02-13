
# TODO (cg 2026/02/06 13:27): for both the primal and the derivatives, the
# evaluation at r=0 is not correct.

using SpectralKernels, ForwardDiff

const ALPHA = inv(50.0)
sdf(w)      = exp(-abs(w)*ALPHA)
kernel(r)   = 2*ALPHA/(ALPHA^2 + (2*pi*r)^2)
cfg         = SpectralKernels.AdaptiveKernelConfig(sdf; tol=1e-12)

xgrid       = vcat([0.0], collect(range(0.001, 1.1, length=10)))
true_values = kernel.(xgrid)
(integrals, errors) = kernel_values(cfg, xgrid; verbose=true)

