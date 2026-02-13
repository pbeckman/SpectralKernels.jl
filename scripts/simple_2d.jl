
using SpectralKernels, StaticArrays 

include("matern_pair.jl")

sdf(w)      = matern_sdf(w, (0.7, 1.3, 1.8); d=2)
kernel(r)   = matern_cov(r, (0.7, 1.3, 1.8); d=2)

cfg         = SpectralKernels.AdaptiveKernelConfig(sdf; dim=2, tol=1e-12)
xgrid       = vcat([0.0], collect(range(0.001, 1.1, length=10)))
true_values = kernel.(xgrid)
(integrals, errors) = kernel_values(cfg, xgrid; verbose=true)

