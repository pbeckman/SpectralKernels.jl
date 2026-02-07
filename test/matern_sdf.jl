
@testset "Matern SDF" begin
  parms = [2.14, 0.97, 0.89]
  xgrid = collect(range(0.001, 5.1, length=1_000))

  for dim in [1, 2]
    @testset "(dim=$(dim))" begin
      S(w)  = matern_sdf(w, parms; d=dim)
      K(r)  = matern_cov(r, parms; d=dim)
      dK(r) = ForwardDiff.derivative(K, r)
      
      for derivative in [false, true]
        @testset "(derivative=$(derivative))" begin
          for tol in (1e-8, 1e-10, 1e-12)
            true_values = derivative ? dK.(xgrid) : K.(xgrid)

            @testset "(tol=$(tol))" begin
              cfg = SpectralKernels.AdaptiveKernelConfig(
                S, dim=dim, tol=tol, derivative=derivative
                )
              (integrals, errors) = kernel_values(cfg, xgrid)
              empirical_errors = abs.(integrals - true_values)./K(0.0)

              @test all(empirical_errors .<= 10*tol)
            end
          end
        end
      end
    end
  end
end

