using SpectralKernels, ForwardDiff, BenchmarkTools, Printf, Plots, Plots.Measures, Random, LinearAlgebra, LaTeXStrings, DelimitedFiles, FINUFFT

include("../../../SpectralKernels.jl/scripts/matern_pair.jl")

function bin_nodes(nodes, bins)
    nodes_binned = zeros(length(bins))

    b = 1
    for node in nodes
        if node < bins[b+1]
            nodes_binned[b] += 1
        else
            nodes_binned[b+1] += 1
            b += 1
        end
    end

    nodes_binned[nodes_binned .== 0] .= NaN

    return nodes_binned
end

BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000

Random.seed!(123)

parms_init = (0.5, 0.55)
parms = (inv(matern_cov(0, (1.0, parms_init...))), parms_init...)
phi, alpha, nu = parms

S(w) = matern_sdf(w, parms)
K(x) = matern_cov(x, parms)

# tolerances to time and plot
tols = [1e-4, 1e-8, 1e-12]

# number of observations N and interpoint distance n 
Ns = round.(Int64, 10 .^ range(1, stop=2, length=3))
# Ns = round.(Int64, 10 .^ range(1, stop=4, length=10))
ns = Int64.((Ns .- 1) .* Ns / 2 .+ 1)

gr(size=(1000,300))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
pls   = Vector{Plots.Plot}(undef, length(tols))
for (i, tol) in enumerate(tols)
    @printf("Timing ε = %.0e\n", tol)

    # determine trapezoidal rule using Barnett 2023 Corr. 6
    l = sqrt(2nu)/(2pi*alpha) 
    h = (1 + l*sqrt(2/nu)*log(3/tol))^(-1)
    m_trap = ceil(
        Int64, 
        (1/(sqrt(pi)*tol))^(1/(2*nu)) * 1.6sqrt(nu)/(pi*h*l)
    )

    # set up our adaptive quadrature
    quadspec = (2^12, 2^4)
    cfg = AdaptiveKernelConfig(
                S,
                tol=tol,
                quadspec=quadspec,
                convergence_criteria=:tails
            )

    timings = fill(NaN, 3, length(Ns))
    for (j, (N, n)) in enumerate(zip(Ns, ns))
        @printf("  N = %i points (n = %i distances)\n", N, n)        
        pts = rand(N)
        xs_unsort = [
            0.0; 
            vcat([[abs(pts[i] - pts[j]) for j=i+1:N] for i=1:N-1]...)
            ]
        perm = sortperm(xs_unsort)
        xs   = xs_unsort[perm]

        # compute and save quadrature nodes and unconverged distances
        rm("nodes.csv", force=true)
        rm("dists.csv", force=true)
        kernel_values(cfg, xs, verbose=false, save_quad=true)
        nodes = collect.(eachrow(readdlm("nodes.csv", ',', Float64)))
        dists = Vector{Vector{Float64}}(undef, size(nodes, 1))
        open("dists.csv", "r") do dists_file
            for i=1:size(nodes, 1)
                dists[i] = parse.(Float64, split(readline(dists_file), ','))
            end
        end
        m_ours = sum(length.(nodes))

        # time our method
        buf = ComplexF64.(randn(prod(cfg.quadspec)))
        timings[1, j] = @belapsed begin
            for (panel_nodes, panel_dists) in zip($nodes, $dists)
                int = nufft1d3(panel_nodes, $buf, +1, 1e-14, 2pi*panel_dists)
            end
        end
        @printf("    our method : %i nodes\t%.2e s\n", m_ours, timings[1, j])

        # time our quadrature scheme but with direct Fourier sums
        if n * m_ours < 1e9
            timings[2, j] = @elapsed begin
                for (panel_nodes, panel_dists) in zip(nodes, dists)
                    int = zeros(ComplexF64, length(panel_dists))
                    for j in eachindex(panel_dists)
                        @inbounds begin
                            xj = panel_dists[j]
                            @simd for k in eachindex(panel_nodes)
                                int[j] += buf[k]*cispi(2*panel_nodes[k]*xj)
                            end
                        end
                    end
                end
            end
        end
        @printf("    direct GL  : %i nodes\t%.2e s\n", m_ours, timings[2, j])

        # time NUFFT accelerated trapezoidal rule
        if m_trap < 1e9
            buf = ComplexF64.(h*S.(-h*m_trap:h:h*m_trap))
            timings[3, j] = @belapsed begin
                int = nufft1d2(2pi*$xs*$h, +1, 1e-14, $buf)
            end
        end
        @printf("    NUFFT trap : %i nodes\t%.2e s\n", 2m_trap+1, timings[3, j])
    end

    # writing timings to file
    open("../timing.csv", "a") do file
        writedlm(file, replace(x -> isnan(x) ? 0 : x, timings))
        write(file, "\n\n")
    end

    # plot timings
    pls[i] = plot(
        ns, timings', 
        title=latexstring(@sprintf("\$\\varepsilon = \\texttt{%.0e}\$", tol)),
        xlabel=L"$n$",
        ylabel="CPU time (s)",
        label="",
        legend=(i==1 ? :topright : :none),
        xscale=:log10, yscale=:log10,
        bottommargin=8mm,
        leftmargin=8mm,
        markersize=4,
        markerstrokewidth=0,
        labels=["NUFFT GL" "direct GL" "NUFFT trap"],
        markershape=[:rect :circle :diamond],
        color=[:red :blue :green],
        linestyle=:dash
        )
    display(pls[i])
end

pl = plot(pls..., layout=grid(1, length(tols)))
display(pl)

# savefig(pl, "../timing.pdf")