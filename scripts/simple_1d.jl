
using SpectralKernels

const ALPHA = inv(50.0)
sdf(w)      = exp(-abs(w)*ALPHA)
cfg         = SpectralKernels.AdaptiveKernelConfig(sdf; tol=1e-12)

xgrid       = vcat([0.0], collect(range(0.001, 1.1, length=1_000)))
true_values = [2*ALPHA/(ALPHA^2 + (2*pi*r)^2) for r in xgrid]
(integrals, errors) = kernel_values(cfg, xgrid; verbose=true)

# relative error w.r.t. K(0) versus error estimate.
display(hcat(abs.(integrals - true_values)[2:end]./true_values[1], errors[2:end]))

