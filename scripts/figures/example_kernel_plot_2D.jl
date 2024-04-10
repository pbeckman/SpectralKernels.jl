
using SpectralKernels, LinearAlgebra, Plots, Plots.PlotMeasures, Printf, LaTeXStrings, Polynomials

type = "oscillatory"

if type == "oscillatory"
    p = 0.0
    v, a, b, c, d = [2.3, 6, 1e-2, 1, 1.3]
    S(w) = (1 + (w/a)^2)^(-v-0.5) * (1 - exp(-w*b)*sin(2pi*c*(w/a)^d))
    # b   = 20
    # cs  = 10 .^ range(0, stop=-12, length=b)
    # w0s = range(0, stop=40, length=b) .^ 1.3
    # ls  = range(w0s[2]/30, stop=w0s[2], length=b)
    # S(w) = sum(c*exp(-abs(abs(w) - w0)/l) for (c, w0, l) in zip(cs, w0s, ls))
elseif type == "chebyshev"
    a, b = [5, 10000]
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
    S(w) = exp(-w/b + ChebyshevT(c)((w-a)/(w+a)))
    p = -0.6
else
    error("kernel options are 'oscillatory' and 'chebyshev'.")
end

# set up adaptive integration config
# if !(@isdefined cfg)
    const tol = 1e-6
    const quadspec = (2^12, 2^0)
    const cfg = AdaptiveKernelConfig(
        S, tol=tol, 
        quadspec=quadspec, 
        convergence_criteria=:panel,
        p=p
        )
# end

N = 91
xs_1D = collect(range(0.0, stop=1.0, length=N))
xs_2D = collect.(Iterators.product(xs_1D, xs_1D))
rs    = sort(unique(norm.(xs_2D)))
@time K_fourier, err_est = kernel_values(cfg, rs, verbose=true, bessel=true);
k0 = K_fourier[1]

dict = Dict(zip(Float32.(rs), K_fourier))

##

xs_plot_1D = xs_1D .- 0.5
xs_plot_2D = collect.(Iterators.product(xs_plot_1D, xs_plot_1D))
K_mat = Matrix{Float64}(undef, N, N)
for i=1:N
    for j=1:N
        K_mat[i,j] = dict[Float32(norm(xs_plot_2D[i,j]))]
    end
end

n = length(xs_2D)
C = Matrix{Float64}(undef, n, n)
for i=1:n
    for j=1:n
        C[i,j] = dict[Float32(norm(xs_2D[i] - xs_2D[j]))]
    end
end
@show extrema(eigvals(C))
W = cholesky(C).L

##

# sample = W*randn(n)

# ws_plot_width = (type=="oscillatory" ? 20 : 200)
ws_plot_width = 40
ws_1D = range(-ws_plot_width, stop=ws_plot_width, length=N)
ws_2D = [norm([ws_1D[j], ws_1D[k]]) for j=1:N, k=1:N]
S_mat = ws_2D.^p .* S.(ws_2D)

gr(size=(1100,230))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
p1 = heatmap(
    ws_1D, ws_1D, log10.(S_mat / k0), 
    # title=L"$S(\omega)$", 
    label="",
    xticks=[-30, 0, 30],
    yticks=[-30, 0, 30],
    # levels=100, linewidth=0, 
    color=:Purples, 
    clims=(-7, maximum(log10.(S_mat / k0))),
    dpi=300,
    xlabel=L"$\omega_1$",
    ylabel=L"$\omega_2$",
    )
cmax = maximum(K_mat) / k0
p2 = heatmap(
    xs_1D, xs_1D, K_mat / k0, 
    # title=L"$K(r)$", 
    label="",
    xticks=(0:0.5:1, ["-0.5", "0", "0.5"]),
    yticks=(0:0.5:1, ["-0.5", "0", "0.5"]),
    # levels=100, linewidth=0, 
    color=reverse(cgrad(:RdBu)), clims=(-cmax, cmax),
    dpi=300,
    xlabel=L"$r_1$",
    ylabel=L"$r_2$"
    )
p3 = heatmap(
    xs_1D, xs_1D, reshape(sample, N, N) / sqrt(k0),
    # title=L"$Z(x)$", 
    label="",
    xticks=(0:0.5:1, ["0", "0.5", "1"]),
    yticks=(0:0.5:1, ["0", "0.5", "1"]),
    # levels=100, linewidth=0, 
    color=reverse(cgrad(:Spectral)),
    dpi=300,
    xlabel=L"$x_1$",
    ylabel=L"$x_2$"
    )

pl = plot(p1, p2, p3, layout=grid(1,3), bottommargin=8mm, leftmargin=10mm, rightmargin=0mm)
display(pl)

##

savefig(pl, "../example_$(type)_2D.pdf")
savefig(pl, "../example_$(type)_2D.png")