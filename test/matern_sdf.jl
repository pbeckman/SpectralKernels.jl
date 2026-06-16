
@testset "Matern SDF" begin
  # ϕ, α, ν
  parms = [2.14, 0.97, 0.89]
  xgrid = collect(range(0.0, 5.1, length=1_000))

  for dim in [1, 2, 4]
    @testset "(dim=$(dim))" begin
      S(w)  = matern_sdf(w, parms; d=dim)
      K(r)  = matern_cov(r, parms; d=dim)
      dK(r) = ForwardDiff.derivative(K, r)
      
      for derivative in [false, true]
        @testset "(derivative=$(derivative))" begin
          i0 = 1 + derivative
          for tol in [1e-4, 1e-8, 1e-10, 1e-12]
            true_values = derivative ? dK.(xgrid[i0:end]) : K.(xgrid[i0:end])

            @testset "(tol=$(tol))" begin
              cfg = SpectralKernels.AdaptiveKernelConfig(
                S, dim=dim, tol=tol, derivative=derivative
                )
              (integrals, errors) = kernel_values(cfg, xgrid[i0:end])
              empirical_errors = abs.(integrals - true_values)./K(0.0)

              @test all(empirical_errors .<= 10*tol)
            end
          end
        end
      end
    end
  end
end

@testset "Singular Matern SDF" begin
  parms = [2.14, 0.97, 0.89]
  xgrid = collect(range(0.0, 1.1, length=1_000))

  for dim in [1, 2, 4]
    alpha = (dim-1) + 0.1
    @testset "(dim=$(dim))" begin
      S(w)  = matern_sdf(w, parms; d=dim)
      K(r)  = sing_matern_cov(r, [parms; -alpha]; d=dim)
      dK(r) = ForwardDiff.derivative(K, r)
      
      for derivative in [false, true]
        @testset "(derivative=$(derivative))" begin
          i0 = 1 + derivative
          for tol in (1e-4, 1e-8, 1e-10, 1e-12)
            true_values = derivative ? dK.(xgrid[i0:end]) : K.(xgrid[i0:end])

            @testset "(tol=$(tol))" begin
              cfg = SpectralKernels.AdaptiveKernelConfig(
                S, dim=dim, tol=tol, derivative=derivative, alpha=alpha
                )
              (integrals, errors) = kernel_values(cfg, xgrid[i0:end])
              empirical_errors = abs.(integrals - true_values)./K(0.0)

              @test all(empirical_errors .<= 10*tol)
            end
          end
        end
      end
    end
  end
end