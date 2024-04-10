using FINUFFT, LinearAlgebra, Plots, Plots.PlotMeasures

include("../../../SpectralKernels.jl/scripts/matern_pair.jl")

# Matern parameters alpha, rho, nu
parms = [1.0, 1.0, 0.5]
# normalize for convenience
parms[1] = inv(matern_cov(0, parms))

# set up Matern Fourier pair
S(w) = matern_sdf(w, parms)
K(x) = matern_cov(x, parms)

# set truncation tolerance for S to define domain of integration
tol = 1e-8
# set discretization width to resolve S
h   = 1e-2
# set left endpoint of integration region
L = 1e+05

# find number of Fourier modes / discretization points in Fourier integral
m = ceil(Int64, L/h)
h = L/(m-1)

# points at which to evaluate K
n = 50
xs = 10 .^ range(-6, stop=0, length=n)

if m > 1e8
    error("You're using a grid of scary large size m = $m... turn down the tolerance.")
else
    println("Using m = $m Fourier modes, x in [$(minimum(xs)), $(maximum(xs))], ω in [0, $L].")
end

# trapezoidal nodes and weights
nds = [(j-1)*h for j=1:m]
wts = fill(h, m)
wts[[1,end]] ./= 2

# compute kernel by brute forcing the Fourier integral
t = @elapsed begin
    # K_naive = 2L / N * real.(fftshift(fft(fftshift(S.(w_grid)))))
    K_naive = 2*real.(vec(nufft1d3(nds, Vector{ComplexF64}(wts .* S.(nds)), +1, 1e-15, 2pi*xs)))
end
err = K.(xs) - K_naive

println("Integration of S to tol = $tol using naive IFFT took $t seconds.")

println("2-norm error = $(norm(err)), inf norm error = $(norm(err, Inf))")

# plot everything
nsk = ceil(Int64, n/1000)
msk = ceil(Int64, m/1000)

gr(size=(700,300))
default(fontfamily="Computer Modern")
ws = range(0, stop=1.3*L, length=1000)
Slims = collect(extrema(S.(ws)))
nds = range(0, stop=L, length=m)
sk = ceil(Int64, m/16)
p1 = plot(
    ws[ws .< L]/1e5, S.(ws[ws .< L]), dpi=300,
    label="", xlabel="ω", ylabel="S(ω)",
    xlims=collect(extrema(ws))/1e5, yscale=:log10, 
    c=:black, linewidth=1, left_margin=3mm
    )
scatter!(p1, 
    nds[1:sk:end]/1e5, S.(nds[1:sk:end]), 
    label="", c=:black, markersize=2
    )
plot!(p1, 
    ws[ws .> L]/1e5, S.(ws[ws .> L]), 
    label="", c=:red, linestyle=:dash, rightmargin=10mm
    )
plot!(p1, [L,L]/1e5, Slims, label="", c=:black, linestyle=:dash)

p2 = plot(
    dpi=200,
    yscale=:log10, xscale=:log10,
    ylabel="Absolute error in K(r)", xlabel="r", ylims=[1e-13, 1e-3]
    )
plot!(p2,
    xs[1:nsk:end], 
    abs.(err[1:nsk:end]),
    label="", 
    linestyle=:solid, linealpha=0.0,
    c=:blue, alpha=0.7, marker=2, markerstrokewidth=0
    )

pl = plot(p1, p2, layout=grid(1,2), bottom_margin=3mm)
display(pl)

savefig(pl, "../trapezoidal.pdf")