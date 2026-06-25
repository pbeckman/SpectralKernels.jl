
@testset "Matern SDF" begin
  # ϕ, ρ, ν
  parms = [2.14, 0.97, 0.89]
  xgrid = collect(range(0.0, 5.1, length=1_000))

  for dim in [1, 2]
    @testset "(dim=$(dim))" begin
      S(w)  = matern_sdf(w, parms; d=dim)
      K(r)  = matern_cov(r, parms; d=dim)
      dK(r) = (r == 0 ? 0.0 : ForwardDiff.derivative(K, r))
      
      for derivative in [false, true]
        @testset "(derivative=$(derivative))" begin
          for tol in [1e-4, 1e-8, 1e-10, 1e-12]
            true_values = derivative ? dK.(xgrid) : K.(xgrid)

            @testset "(tol=$(tol))" begin
              cfg = SpectralKernels.AdaptiveKernelConfig(
                S, dim=dim, tol=tol, derivative=derivative
                )
              (integrals, errors) = kernel_values(
                cfg, xgrid; (derivative ? (; k0=K(0.0)) : (;))...
                )
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

  for dim in [1, 2]
    alpha = (dim-1) + 0.5
    @testset "(dim=$(dim))" begin
      S(w)  = matern_sdf(w, parms; d=dim)
      dS(w) = ForwardDiff.derivative(S, w)
      K(r)  = sing_matern_cov(r, [parms; -alpha]; d=dim)
      dK(r) = (r == 0 ? 0.0 : ForwardDiff.derivative(K, r))
      
      for derivative in [false, true]
        @testset "(derivative=$(derivative))" begin
          for tol in (1e-4, 1e-8, 1e-10, 1e-12)
            true_values = derivative ? dK.(xgrid) : K.(xgrid)

            @testset "(tol=$(tol))" begin
              cfg = SpectralKernels.AdaptiveKernelConfig(
                S, dim=dim, tol=tol, derivative=derivative, alpha=alpha
                )
              (integrals, errors) = kernel_values(
                cfg, xgrid; (derivative ? (; k0=K(0.0)) : (;))...
                )
              empirical_errors = abs.(integrals - true_values)./K(0.0)

              @test all(empirical_errors .<= 10*tol)
            end
          end
        end
      end

      @testset "derivative in α" begin
        Kda(r) = ForwardDiff.derivative(
          a -> sing_matern_cov(r, [parms; -a]; d=dim), 
          alpha
          )
        true_values = Kda.(xgrid)

        cfg = SpectralKernels.AdaptiveKernelConfig(
          S, df=dS, dim=dim, tol=tol, alpha=alpha, logw=true
          )
        (integrals, errors) = kernel_values(
          cfg, xgrid, param_derivative=true, k0=K(0.0)
          )
        empirical_errors = abs.(integrals - true_values)./K(0.0)

        @test all(empirical_errors .<= 10*tol)
      end
    end
  end
end