
using SpectralKernels

const ALPHA = inv(50.0)
sdf(w)      = exp(-abs(w)*ALPHA)
truth(r)    = 2*ALPHA/(ALPHA^2 + (2*pi*r)^2)

integrand = SpectralKernels.Integrand(sdf)
cfg       = SpectralKernels.AdaptiveKernelConfig(;tol=1e-12)

xgrid       = collect(range(0.001, 1.1, length=1_000))
true_values = truth.(xgrid)
(integrals, errors) = kernel_values(integrand, xgrid, cfg; verbose=true)

# relative error w.r.t. K(0) versus error estimate.
display(hcat(abs.(integrals - true_values)./truth(0.0), errors))

