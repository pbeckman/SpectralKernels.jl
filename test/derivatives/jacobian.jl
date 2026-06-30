
@testset "Jacobian" begin

  # specify the model and the kernel, and then the warping function:
  iso_sdf(w, alpha)    = exp(-alpha*abs(w))
  iso_kernel(r, alpha) = 2*alpha/(alpha^2 + (2*pi*r)^2)
  warp(params, x)      = x^params[1] 

  kernel(x, y, params) = iso_kernel(abs(warp((params[2],), x) - warp((params[2],), y)), params[1])

  test_params = [1.1, 0.1]
  backend     = AutoForwardDiff()

  cfg    = SpectralKernels.AdaptiveKernelConfig(iso_sdf; tol=1e-12)
  xgrid  = collect(range(0.0, 1.0, length=20))

  model = SpectralModel(iso_sdf, xgrid; warp, sdf_param_indices=1, 
                        warp_param_indices=2, verbose=false)

  k0 = SpectralKernels.compute_k0(cfg; params=test_params[1])
  J_test = ForwardDiff.jacobian(
    params -> begin
      gk = gen_kernel(model, params)
      [gk(xy[1], xy[2], params) for xy in Iterators.product(xgrid, xgrid)]
    end,
    test_params)

  J_ref = ForwardDiff.jacobian(
    params -> [kernel(xy[1], xy[2], params) for xy in Iterators.product(xgrid, xgrid)],
    test_params
    )

  @test maximum(abs, J_test - J_ref) < 1e-8

end

