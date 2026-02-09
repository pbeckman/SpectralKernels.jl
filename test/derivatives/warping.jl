
@testset "warping" begin

  # specify the model and the kernel, and then the warping function:
  iso_sdf(w)        = exp(-abs(w))
  iso_kernel(r)     = 2/(1 + (2*pi*r)^2)
  warp(params, x)   = (x/params[1])^params[2]
  kernel(x, y, params) = iso_kernel(norm(warp(params, x) - warp(params, y)))

  test_params = [inv(50.0), 1.1]
  backend     = AutoForwardDiff()

  cfg    = SpectralKernels.AdaptiveKernelConfig(iso_sdf; tol=1e-12)
  xpairs = [(1.0, x) for x in range(1.1, 2.0, length=100)]

  manual_warp_lags = [norm(warp(test_params, xp[1]) - warp(test_params, xp[2]))
                      for xp in xpairs]

  # test 1: this warping function representation is coded and functioning as
  # expected.
  kv     = kernel_values(cfg, manual_warp_lags; verbose=false)[1]
  should = [kernel(xp..., test_params) for xp in xpairs]
  @test kv ≈ should

  # Test 2: the gradient of the warping function at each location can be computed
  # with the DI interface.
  test_grads = SpectralKernels.warping_gradients(warp, xpairs, test_params; backend=backend)
  warp_grads = map(xpairs) do xy
    (x, y) = xy
    ForwardDiff.gradient(p->norm(warp(p, x) - warp(p, y)), test_params)
  end
  @test test_grads == warp_grads 

  # Test 3: derivatives of the kernel with respect to warping parameters.
  test_grads   = SpectralKernels.kernel_warping_gradients(cfg, warp, xpairs, 
                                                          manual_warp_lags,
                                                          test_params;
                                                          backend=backend)
  kernel_grads = map(xpairs) do xy
    (x, y) = xy
    ForwardDiff.gradient(p->kernel(x, y, p), test_params)
  end
  @test maximum(norm.(test_grads .- kernel_grads)) < 1e-8

end

