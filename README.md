# SpectralKernels.jl

This package uses adaptive Gaussian quadrature accelerated by the nonuniform
fast Fourier transform (NUFFT) to compute Fourier transforms of continuous,
integrable functions. 

In the Gaussian process context for which this code was developed, given a
positive even spectral density $S(\omega)$ we wish to compute the corresponding
covariance function
```math
K(r) 
:= \int_{-\infty}^\infty S(\omega) e^{2\pi i\omega r} \, d\omega
= 2\int_0^\infty S(\omega) \cos(2\pi\omega r) \, d\omega
```
at a collection of distances $r_1, \dots, r_n \in \mathbb{R}$ to a
user-specified tolerance $\varepsilon$. 

A minimal demo is as follows:
```julia
using SpectralKernels

# distances r at which to evaluate K
n  = 1_000_000
rs = 10 .^ range(-6, stop=0, length=n) 

# specify an integrable spectral density
S(w) = (1 + w^2)^(-2)

# set up adaptive integration config
cfg = AdaptiveKernelConfig(S)

# compute covariance function K(r) for all distances r
Ks = kernel_values(cfg, rs)
```
See the `scripts` directory for more detailed, heavily commented demos. 

# Advanced Usage

## NUFFT multithreading

The speed of this package is due largely to the NUFFT. For this purpose we use
the [FINUFFT library](https://finufft.readthedocs.io/), whose multi-threading
behavior is determined by the environment variable `OMP_NUM_THREADS`. We
recommend setting this variable to the number of cores on your machine, for
example
```bash
export OMP_NUM_THREADS=8
``` 
on an 8-core laptop. 

## Configuration options

In order to tune accuracy and speed, the `AdaptiveKernelConfig` object accepts
a number of keyword arguments with the following defaults
```julia
AdaptiveKernelConfig(
    S; 
    tol::Float64=1e-10, 
    convergence_criteria::Symbol=:both,
    quadspec::Tuple{Int64, Int64}=(2^12, 2^4)
    )
```

`tol` : pointwise error tolerance $|K(r) -
\widetilde{K}(r)| \ / \ K(0) < \varepsilon$

`convergence_criteria` :

`quadspec` :

## Singular spectral densities

