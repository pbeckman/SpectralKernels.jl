using SpectralKernels, ForwardDiff, BenchmarkTools, Printf, Plots, Plots.Measures, Random, LinearAlgebra, LaTeXStrings

include("../../../SpectralKernels.jl/scripts/matern_pair.jl")

alpha = 0.1
parms_init = (0.5, 0.51)
parms = (inv(matern_cov(0, (1.0, parms_init...))), parms_init...)

quadspec = (2^12, 2^0)

Random.seed!(123)
npts = 1000
pts  = sort(rand(npts))

tols = 10.0 .^ (-6:-1:-12)
errs = Matrix{Float64}(undef, 6, length(tols))

dsdfv = SpectralKernels.component_derivatives(matern_sdf, Val(4))
nusdf = SpectralKernels.ParametricFun(dsdfv[3], tuple(parms...))
psdf  = SpectralKernels.ParametricFun(matern_sdf, tuple(parms...))

gr(size=(1050,290))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
pls  = Vector{Plots.Plot}(undef, 4)
iter = zip(
    [L"K(r), \alpha=0", 
    L"K(r)", 
    L"\frac{\partial K(r)}{\partial\nu}", 
    L"\frac{\partial K(r)}{\partial \alpha}"],
    [
        x -> matern_cov(x, parms), 
        x -> sing_matern_cov(x, p, parms), 
        x -> ForwardDiff.derivative(_nu->sing_matern_cov(x, p, [parms[1:2]...; _nu]), parms[3]),
        x -> ForwardDiff.derivative(_p->sing_matern_cov(x, _p, parms), p)
        ],
    [
        w -> matern_sdf(w, parms), 
        w -> matern_sdf(w, parms), 
        nusdf, 
        psdf],
    [0, alpha, alpha, alpha],
    [false, false, false, true]
)
for (j, (title, K, S, alpha, logw)) in enumerate(iter)
    @printf("Making plot %i\n", j)
    M_kernel = [K(abs(p1 - p2)) for p1 in pts, p2 in pts]
    for (i, tol) in enumerate(tols)
        @printf("-- computing errors for tol = %.2e\n", tol)
        
        cfg = AdaptiveKernelConfig(
            S,
            tol=tol, 
            quadspec=quadspec, 
            alpha=alpha,
            logw=logw
        )

        M_fourier = SpectralKernels.build_dense_cov_matrix(cfg, pts)

        errs[1,i] = maximum(abs.(M_kernel - M_fourier))
        errs[2,i] = opnorm(M_kernel - M_fourier)
        errs[3,i] = norm(M_kernel - M_fourier) 

        errs[4,i] = errs[1,i] / maximum(abs.(M_kernel))
        errs[5,i] = errs[2,i] / opnorm(M_kernel)
        errs[6,i] = errs[3,i] / norm(M_kernel)
    end

    pl = plot(
        [1e-16, 1e-4], [1e-16, 1e-4],
        title=title,
        xlabel="ε", 
        ylabel=(j==1) ? "Relative error" : "",
        label="ε",
        xscale=:log10, yscale=:log10,
        xlims=[0.2e-12, 2e-6], 
        xticks=[1e-12, 1e-10, 1e-8, 1e-6],
        ylims=[0.2e-12, 8e-6],
        yticks=[1e-12, 1e-10, 1e-8, 1e-6],
        legend=(j==1) ? :topleft : :none,
        legendfontsize=10,
        markersize=0,
        color=:black,
        bottommargin=10mm,
        leftmargin=(j==1) ? 8mm : 0mm
        )
    plot!(pl,
        tols, errs[4:6,:]', 
        labels=["max" "spectral" "Frobenius"],
        markersize=4,
        markershape=[:rect :circle :diamond],
        markerstrokewidth=0,
        color=[:red :blue :green],
        linestyle=:dash
    )

    pls[j] = pl
end

pl = plot(pls..., layout=grid(1, 4))
display(pl)

savefig(pl, "../relative_error.pdf")