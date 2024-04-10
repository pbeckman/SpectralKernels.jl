# SpectralKernels.jl

This package uses adaptive Gaussian quadrature accelerated by the nonuniform
fast Fourier transform (NUFFT) to compute Fourier transforms of continuous,
integrable functions. In the Gaussian process context for which this code is
designed, given a positive spectral density $S(\omega)$, we wish to
compute the corresponding covariance function given by its inverse Fourier transform
$$ K(r) := \int_{-\infty}^\infty S(\omega) e^{2\pi i\omega r} \, d\omega $$
at a set of distances $r_1, \dots, r_n$.

A minimal demo is as follows:
```julia
using SpectralKernels

# distances r at which to evaluate K
n  = 1_000_000
rs = 10 .^ range(-6, stop=0, length=n) 

# specify an integrable spectral density
S(w) = (1 + w^2)^(-2)

# set up adaptive integration config
cfg = AdaptiveKernelConfig(S, tol=1e-10)

# compute covariance function K(r) for all distances r
Ks = kernel_values(cfg, rs)
```
See the `scripts` directory for more detailed heavily commented demos. 