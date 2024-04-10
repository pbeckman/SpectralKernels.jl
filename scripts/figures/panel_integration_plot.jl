
using SpectralKernels, Plots, Plots.PlotMeasures, Printf, LaTeXStrings

include("../../../SpectralKernels.jl/scripts/matern_pair.jl")

# Matern parameters lengthscale, smoothness
parms_init = (1.0, 0.5)

# normalize so that K(0) = 1
parms = (inv(matern_cov(0, (1.0, parms_init...))), parms_init...)

# set up Matern Fourier pair
S(w) = matern_sdf(w, parms)
K(x) = matern_cov(x, parms)

# set up adaptive integration config
tol = 1e-8
m   = 5000
cfg = AdaptiveKernelConfig(
    S, 
    tol=tol, 
    quadspec=(m, 1),
    tail=ceil(Int64, -2*(parms[3]+0.5))
    )

# compute kernel
n  = 50
xs = 10 .^ range(-6, stop=0, length=n)
K_true = K.(xs)

# setup grid in w space for plotting
ws = range(0, stop=2.2e+05, length=1000)
# ws = 10 .^ range(-8, stop=6, length=1000)
Slims = collect(extrema(S.(ws)))

# integrate some panels
gr(size=(1000,450))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
pls = Matrix{Plots.Plot}(undef, 4, 2)
K_fourier = zeros(length(xs))
is_unconverged = trues(length(xs))
(a, b) = (0, 0)
for j=1:4
    highest_unconv_ix = findlast(is_unconverged)
    (a, b) = (b, b + SpectralKernels.quadsz(cfg) / (2*xs[highest_unconv_ix]))
    @show sum(is_unconverged)
    @show b-a
    xs_unconv = xs[is_unconverged]
    view(K_fourier, is_unconverged) .+= 2*SpectralKernels.fourier_integrate_interval(a, b, cfg, xs_unconv, 1.0, true)[1]
    is_unconverged[abs.(K_fourier - K_true) .< tol] .= false
    
    # plot spectral density
    nds = a .+ (cfg.legrule.no1 .+ 1)*(b-a)/2
    sk = ceil(Int64, m/16)
    pls[j,1] = plot(
        ws[ws .< a], cfg.sdf.(ws[ws .< a]), dpi=300,
        title=latexstring(@sprintf("(b-a) = \\texttt{%.1e}", b-a)),
        titlefontsize=16,
        label="", xlabel=L"$\omega$", ylabel=((j==1) ? L"$S(\omega)$" : ""),
        xlims=collect(extrema(ws)), 
        yscale=:log10,
        xticks=([0, 1e5, 2e5],["0", L"10^5", L"2 \times 10^5"]),
        c=:blue, linewidth=2, left_margin=((j==1) ? 5mm : 0mm), right_margin=((j==4) ? 3mm : 0mm)
        )
    plot!(pls[j,1], 
        ws[(ws .> a) .* (ws .< b)], cfg.sdf.(ws[(ws .> a) .* (ws .< b)]), 
        label="", c=:black, linewidth=1
        )
    scatter!(pls[j,1], 
        nds[1:sk:end], cfg.sdf.(nds[1:sk:end]), 
        label="", c=:black, markersize=2
        )
    plot!(pls[j,1], 
        ws[ws .> b], cfg.sdf.(ws[ws .> b]), 
        label="", c=:red, linestyle=:dash
        )
    plot!(pls[j,1], [a,a], Slims, label="", c=:black, linestyle=:dash)
    plot!(pls[j,1], [b,b], Slims, label="", c=:black, linestyle=:dash)

    # plot errors
    pls[j,2] = scatter(
        xs[is_unconverged], abs.(K_true - K_fourier)[is_unconverged], dpi=300,
        label="", xlabel=L"$r$", ylabel=((j==1) ? L"Abs. err. in $K(r)$" : ""),
        yscale=:log10, xscale=:log10, ylims=[1e-13, 1e-3],
        xticks=10. .^ (-6:2:0), 
        markerstrokecolor=:red, markersize=2, 
        markercolor=:white, markerstrokewidth=1,
        left_margin=((j==1) ? 5mm : 0mm), bottom_margin=5mm,
        top_margin=8mm
        )
    scatter!(pls[j,2],
        xs[broadcast(!, is_unconverged)], abs.(K_true - K_fourier)[broadcast(!, is_unconverged)], 
        label="",
        markersize=2, color=:green, markerstrokewidth=0
        )
    plot!(pls[j,2], collect(extrema(xs)), [tol, tol], label="", c=:black, linestyle=:dash)
end

pl = plot(pls..., layout=grid(2,4))
display(pl)

savefig(pl, "../panel_integration.pdf")