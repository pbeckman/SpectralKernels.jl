
using SpectralKernels, LinearAlgebra, Plots, Plots.PlotMeasures, Printf, LaTeXStrings

# Constraints:
# alpha*(nu + 1/2) - beta > 1.0

function S(w) 
  (phi, lam, beta, rho, alpha, nu) = (1.0, 0.01, 0.75, 1.0, 1.65, 1.45)
  phi*(lam + (1-lam)*abs(w)^beta)*(rho^2 + abs(w)^alpha)^(-nu - 1/2)
end

# set up adaptive integration config
const n   = 1000
const xs  = collect(range(0.0, stop=1.0, length=n))
const cfg = AdaptiveKernelConfig(S, tol=1e-12, quadspec=(2^14, 2^4),
                                 convergence_criteria=:panel)
const K_fourier, err_est = kernel_values(cfg, xs, verbose=true);
K_fourier ./= K_fourier[1]

function gencov()
  C = [K_fourier[abs(i-j)+1] for i in 1:n, j in 1:n]
  @show extrema(eigvals(C))
  W = cholesky(C).L
  samples = W*randn(n, 1)
  (C, samples)
end
(C, samples) = gencov()

ws = range(0, stop=500, length=n)[2:end]

gr(size=(1000,200))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)
p1 = plot(
    title=L"$S(\omega)$",
    ws, S.(ws), label="",
    linewidth=1, color=:blue,
    # xscale=:log10,
    yticks=[1e-6, 1e-3],
    yscale=:log10,
    dpi=200,
    xlabel=L"$\omega$")
p2 = plot(
    title=L"$K(r)$",
    xs, K_fourier, label="",
    linewidth=1, color=:red,
    dpi=200,
    xlabel=L"$r$")
p3 = plot(
    title=L"$Z(x)$",
    xs, samples, label="",
    linewidth=1, color=:black,
    dpi=200,
    xlabel=L"$x$")

pl = plot(p1, p2, p3, layout=grid(1,3), bottommargin=6mm)
display(pl)

savefig(pl, "../example_generalizedmatern.pdf")

