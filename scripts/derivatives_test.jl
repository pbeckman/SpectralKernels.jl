
using SpectralKernels, DifferentiationInterface, ForwardDiff

include("matern_pair.jl")

# Note the SDF is written in scalar code!
iso_matern_sdf(w, p1, p2, p3) = p1*(p2^2 + w^2)^(-p3 - 1/2)
psdf = SpectralKernels.ParametricFunction(iso_matern_sdf, (2.3, 0.1, 1.75))

cfg    = AdaptiveKernelConfig(psdf; tol=1e-12)

xs     = collect(range(0.01, 3.5, length=30))
derivs = SpectralKernels.kernel_sdf_derivatives(cfg, xs; backend=AutoForwardDiff())

manual_derivs1 = [ForwardDiff.derivative(p1->matern_cov(r, [p1, 0.1, 1.75]), 2.3)
                  for r in xs]
manual_derivs2 = [ForwardDiff.derivative(p2->matern_cov(r, [2.3, p2, 1.75]), 0.1)
                  for r in xs]
manual_derivs3 = [ForwardDiff.derivative(p3->matern_cov(r, [2.3, 0.1, p3]), 1.75)
                  for r in xs]

@show maximum(abs.(derivs[1] .- manual_derivs1)) < 1e-5
@show maximum(abs.(derivs[2] .- manual_derivs2)) < 1e-5
@show maximum(abs.(derivs[3] .- manual_derivs3)) < 1e-5

