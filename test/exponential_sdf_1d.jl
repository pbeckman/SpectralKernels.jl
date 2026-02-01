
@testset "1D Exponential SDF" begin
  S(w)  = exp(-abs(w))
  K(r)  = 2 / (1 + (2*pi*r)^2)
  dK(r) = -(16pi^2*r) / (1 + (2*pi*r)^2)^2
  
  xgrid = collect(range(0.001, 5.1, length=1_000))

  for derivative in [false, true]
    @testset "(derivative=$(derivative))" begin
      for tol in (1e-8, 1e-10, 1e-12)
        true_values = derivative ? dK.(xgrid) : K.(xgrid)
        @testset "(tol=$(tol))" begin
          cfg = SpectralKernels.AdaptiveKernelConfig(
            S, tol=tol, derivative=derivative
            )
          (integrals, errors) = kernel_values(cfg, xgrid)
          empirical_errors = abs.(integrals - true_values)./K(0.0)
          @test all(empirical_errors .<= 10*tol)
        end
      end
    end
  end
end
