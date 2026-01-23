
@testset "1D Exponential SDF" begin
  sdf(w)      = exp(-abs(w)/50.0)
  truth(r)    = (2/50.0)/(inv(50.0)^2 + (2*pi*r)^2)
  integrand   = SpectralKernels.Integrand(sdf)

  for tol in (1e-8, 1e-10, 1e-12)
    @testset "(tol=$(tol))"  begin
      cfg         = SpectralKernels.AdaptiveKernelConfig(;tol=tol)
      xgrid       = collect(range(0.001, 1.1, length=1_000))
      true_values = truth.(xgrid)
      (integrals, errors) = kernel_values(integrand, xgrid, cfg)
      empirical_errors = abs.(integrals - true_values)./truth(0.0)
      @test all(empirical_errors .<= tol*3)
    end
  end

end

