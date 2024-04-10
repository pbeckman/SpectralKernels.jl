
using SpectralKernels, Printf, ForwardDiff, LinearAlgebra, Plots
include("matern_pair.jl")

# singularity parameter (set to 0 for standard Matern)
const alpha = 0.2

# Matern parameters lengthscale and smoothness
const parms_init = (10.0, 0.5)

# choose standard or singular Matern
if alpha == 0
    kernel(r, parms) = matern_cov(r, parms)
else
    kernel(r, parms) = sing_matern_cov(r, -alpha, parms)
end

# normalize so that K(0) = 1
const parms = (inv(kernel(0, (1.0, parms_init...))), parms_init...)

# set up Matern Fourier pair with given parameters
S(w) = matern_sdf(w, parms)
K(r) = kernel(r, parms)

# desired accuracy
const tol = 1e-10
# (m, k) uses k-many m-node panels
# keyword can be left out and a well-tuned default is used
const quadspec = (2^8, 2^4)

# set up adaptive integration config
const cfg = AdaptiveKernelConfig(
    S, 
    p=-alpha,
    tol=tol,
    quadspec=quadspec
    )

# choose distances r at which to evaluate K
n  = 1_000_000
xs = 10 .^ range(-8, stop=-2, length=n) 

# adaptively compute kernel values
@time K_fourier, err_est = kernel_values(cfg, xs, verbose=true);

# compute analytic kernel values, but beware that K (not S) is unstable 
# for p ≠ 0 at even moderate r
K_true   = K.(xs)
err_true = K_true - K_fourier

@printf("\naverage absolute error: %.2e\n", sum(abs.(err_true)) / n)
@printf("max absolute error:     %.2e\n\n", maximum(abs.(err_true)))

# skip points when plotting many points to avoid slowness
sk = ceil(Int64, n/200)
gr(size=(1000,1000))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
pl = plot(
    title=@sprintf(
        "ν = %.2f, α = %.1f, tol = %.1e", 
        parms[3], alpha, tol
        ),
    legend=:bottomright,
    dpi=200,
    yscale=:log10, 
    xscale=:log10, 
    xlabel="x", ylims=[1e-16, 1e0]
    )
plot!(pl,
    xs[1:sk:end], 
    abs.(err_est[1:sk:end]),
    label="absolute error estimate", 
    linestyle=:dash, 
    linecolor=:red,
    linealpha=0.3,
    markerstrokecolor=:red, 
    markercolor=:white, 
    markersize=3,
    markershape=:circle
    )
plot!(pl,
    xs[1:sk:end], 
    abs.(err_true[1:sk:end]),
    label="absolute error", 
    linestyle=:solid, linealpha=0.2,
    c=:blue, alpha=0.7, marker=2, markerstrokewidth=0
    )
plot!(pl,
    [extrema(xs)...], 
    [tol, tol],
    label="tolerance", linestyle=:dash, linewidth=2, c=:blue
    )
display(pl)
