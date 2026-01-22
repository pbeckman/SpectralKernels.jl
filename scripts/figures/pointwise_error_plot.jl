
using SpectralKernels, HypergeometricFunctions, Printf, ForwardDiff, LinearAlgebra, Plots, Plots.Measures, LaTeXStrings
import SpecialFunctions: gamma, besselj0

include("../../../SpectralKernels.jl/scripts/matern_pair.jl")

# dimension
d = 2

# singularity parameter
const alpha = 0.0
type = iszero(alpha) ? "nonsingular" : "singular"

# Matern parameters lengthscale, smoothness
parms_init = (1.0, 0.5 + alpha/2)

function kernel(x, alpha, parms)
    if alpha == 0
        return matern_cov(x, parms, d=d)
    else
        v, a, b = parms
        return v * a^(-2b)*(2pi)^alpha / gamma(0.5 + b) * (
            2^(2b+1)*pi^(2b)*(a*x)^(2b)*x^alpha*cos(b*pi+alpha*pi/2)*gamma(0.5+b)*gamma(-2b-alpha) * 
            pFq((0.5+b,), (0.5+b + alpha/2, 1+b + alpha/2), a^2*pi^2*x^2) + 
            (2pi)^(-alpha)*(a)^(-alpha)*gamma(b + alpha/2)*gamma(0.5 - alpha/2) * 
            pFq((0.5-alpha/2,), (0.5, 1-b-alpha/2), a^2*pi^2*x^2)
        )
    end
end

# normalize so that K(0) = 1
const parms = (inv(kernel(1e-200, alpha, (1.0, parms_init...))), parms_init...)

# set up Matern Fourier pair
S(w) = matern_sdf(w, parms, d=d)
K(r) = kernel(r, alpha, parms)

# points for plotting
n  = 100
rs = 10 .^ range(-8, stop=0, length=n)

tols = [1e-4, 1e-8, 1e-12]
gr(size=(1000,300))
default(fontfamily="Computer Modern")
pls  = Vector{Plots.Plot}(undef, 3)
for (j, tol) in enumerate(tols)
    # set up adaptive integration config if !(@isdefined cfg)
        cfg = AdaptiveKernelConfig(
            S,
            tol=tol,
            convergence_criteria=:both,
            quadspec=(2^8, 2^0), 
            alpha=alpha,
            dim=d
            )
    # end

    @time K_fourier, err_est = kernel_values(cfg, rs, verbose=true);

    K_true   = K.(rs)
    err_true = K_true - K_fourier

    pl = plot(
        legend=(j==1 ? :bottomright : :none),
        dpi=200,
        yscale=:log10, 
        xscale=:log10, 
        xlabel=L"$r$", ylims=[1e-16, 1e-2]
        )
    plot!(pl,
        rs, 
        abs.(err_est),
        label="error estimate", 
        linestyle=:dash, 
        linecolor=:blue,
        linealpha=0.0,
        markerstrokecolor=:blue, 
        markeralpha=0.5,
        markercolor=:white, 
        markersize=3,
        markershape=:circle
        )
    plot!(pl,
        rs, 
        abs.(err_true),
        label="error", 
        linestyle=:solid, linealpha=0.0,
        c=:green, alpha=1, marker=2, markerstrokewidth=0
        )
    plot!(pl,
        [extrema(rs)...], 
        [tol, tol],
        label=L"$\varepsilon$", linestyle=:dash, linewidth=2, c=:black
        )
    pls[j] = pl
end

pl = plot(pls..., layout=grid(1,3), bottommargin=5mm)
display(pl)

savefig(pl, "../pointwise_error_$type.pdf")

##

ws = range(0, 1000, 10_000)
h  = ws[end] / (length(ws) - 1)

K_trap = 2pi*h*[sum(S.(ws) .* besselj0.(2pi*ws*r) .*  ws) for r in rs]
