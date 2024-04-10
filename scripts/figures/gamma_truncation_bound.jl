using SpecialFunctions, Plots, Plots.PlotMeasures, LaTeXStrings, Printf

### Gamma function bound

ys   = 10 .^ range(-14, stop=12, length=1000)
s    = -1
vals = abs.(gamma.(s, im*ys))

ref = true
gr(size=(750,250))
default(fontfamily="Computer Modern")
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)
p1 = plot(
    yscale=:log10, scale=:log10, legend=:bottomleft,
    legendfontsize=10,
    # title=L"$|\Gamma(s, iy)|$" * ", s = $s",
    xlabel="y", ylabel="", dpi=300, ylims=[1e-14, 1e13]
)
plot!(p1, 
    ys, vals,
    label=L"$|\Gamma(-s, iy)|$", 
    linewidth=2, c=palette(:default)[4], yticks=[1e-8, 1e0, 1e8]
)
if ref
    plot!(p1, 
        ys, ys.^(s-1),
        label=L"$y^{s-1}$",
        line=(1, :dash, :black)
    )
    plot!(p1,
        ys, -ys.^s/s,
        label=L"$-y^s/s$",
        line=(1, :dashdot, :black)
    )
end

bound = min.(ys.^(s-1), -ys.^s/s)
@printf(
    "min and max relative gap between bound and value (should be positive): (%.2e, %.2e)\n", 
    extrema((bound - vals) ./ vals)...
    )

display(p1)

### Truncation error bound

trunc(d, x, b) = b^(d+1) * real(expint(-d, 2pi*im*b*x))

n  = 100
bs = 10 .^ range(-4, stop=16, length=1000)

ref = true
d  = s-1
xs = 10 .^ range(-6, stop=0, length=3)
p2 = plot(
    yscale=:log10, scale=:log10, legend=:topright,
    legendfontsize=9,
    # title=L"$|\int {}_b^\infty \omega^d \cos(2\pi\omega x) d\omega|$" * ", d = $d",
    xlabel="b", ylabel="", dpi=300, ylims=[1e-14, 1e7]
)
for (j, x) in enumerate(xs)
    plot!(p2, 
        bs, abs.(trunc.(d, x, bs)),
        label=@sprintf("r=%.0e", x),
        c=palette(:default)[j], linewidth=2
    )
    if ref
        plot!(p2, 
            bs, (bs).^d / (2pi*x),
            label="",
            line=(1, :dash, palette(:default)[j])
        )
    end
end
plot!(p2,
    [], [],
    label=L"$b^{-\beta}/(2\pi r)$",
    line=(1, :dash, :black)
)
plot!(p2,
    bs, -bs.^(d+1)/(d+1),
    label=L"$b^{-\beta+1}/(\beta-1)$",
    line=(1, :dashdot, :black)
)

display(p2)

pl = plot(p1, p2, layout=grid(1,2), bottommargin=7mm)

savefig(pl, "../truncation_bound.pdf")