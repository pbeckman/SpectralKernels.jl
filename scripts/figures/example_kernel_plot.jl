
using SpectralKernels, LinearAlgebra, Plots, Plots.PlotMeasures, Printf, LaTeXStrings, Polynomials

type = "oscillatory"

if type == "oscillatory"
    p = 0.0
    v, a, b, c, d = [1.3, 10, 0.01, 1, 1]
    S(w) = (1 + (w/a)^2)^(-v-0.5) * (1 - exp(-w*b)*sin(2pi*c*(w/a)^d))
elseif type == "chebyshev"
    a, b = [10, 100]
    c = [  
        0.9579871715982113
        0.1685655989357389
        -0.5437741543492275
        -0.7869476456654225
        0
        -0.9270253459354798
        -0.9597144380129826
        -0.8243803056618717
        -5
        ]
    S(w) = exp(-w/b) * exp(ChebyshevT(c)((w-a)/(w+a)))
    p = -0.9
else
    error("kernel options are 'oscillatory' and 'chebyshev'.")
end

# set up adaptive integration config
# if !(@isdefined cfg)
    const tol = 1e-12
    const quadspec = (2^12, 2^0)
    const cfg = AdaptiveKernelConfig(
        S, tol=tol, 
        quadspec=quadspec, 
        convergence_criteria=:panel,
        p=p
        )
# end

n = 1000
xs = collect(range(0.0, stop=1.0, length=n))
@time K_fourier, err_est = kernel_values(cfg, xs, verbose=true);
k0 = K_fourier[1]

C = Matrix{Float64}(undef, n, n)
for i=1:n
    for j=1:n
        C[i,j] = K_fourier[abs(i-j)+1]
    end
end
@show extrema(eigvals(C))
W = cholesky(C + 1e-10*Diagonal(ones(n))).L
#

samples = W*randn(n, 1)

#

ws = range(0, stop=500, length=n)[2:end]

gr(size=(1000,200))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.25)
p1 = plot(
    title=(type == "oscillatory" ? L"$S(\omega)$" : ""),
    ws, ws.^p .* S.(ws) / k0, label="",
    linewidth=1, color=:blue,
    # xscale=:log10,
    yscale=:log10,
    dpi=200,
    xlabel=(type == "oscillatory" ? "" : L"$\omega$")
    )
p2 = plot(
    title=(type == "oscillatory" ? L"$K(r)$" : ""),
    xs, K_fourier / k0, label="",
    linewidth=1, color=:red,
    yticks=[-0.25, 0, 0.25, 0.5, 0.75, 1],
    dpi=200,
    xlabel=(type == "oscillatory" ? "" : L"$r$")
    )
p3 = plot(
    title=(type == "oscillatory" ? L"$Z(x)$" : ""),
    xs, samples / sqrt(k0), label="",
    linewidth=1, color=:black,
    dpi=200,
    xlabel=(type == "oscillatory" ? "" : L"$x$")
    )

pl = plot(p1, p2, p3, layout=grid(1,3), bottommargin=6mm)
display(pl)

savefig(pl, "../example_$type.pdf")