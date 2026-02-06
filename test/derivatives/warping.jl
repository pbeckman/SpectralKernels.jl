
@testset "warping" begin

  # specify the model and the kernel, and then the warping function:
  iso_sdf(w)        = exp(-abs(w))
  iso_kernel(r)     = 2/(1 + (2*pi*r)^2)
  warp(params, x)   = abs((x/params[1])^params[2])
  kernel(r, params) = iso_kernel(warp(params, r))

  test_params = [inv(50.0), 1.1]
  backend     = AutoForwardDiff()

  cfg   = SpectralKernels.AdaptiveKernelConfig(iso_sdf; tol=1e-12)
  xgrid = vcat(collect(range(0.001, 1.1, length=20)))

  # test 1: this warping function representation is coded and functioning as
  # expected.
  kv = kernel_values(cfg, warp.(Ref(test_params), xgrid); verbose=false)
  @test isapprox(kv[1], kernel.(xgrid, Ref(test_params)); rtol=1e-10) # true

  # Test 2: the gradient of the warping function at each location can be computed
  # with the DI interface.
  test_grads = SpectralKernels.warping_gradients(warp, xgrid, test_params; backend=backend)
  warp_grads = [ForwardDiff.gradient(p->warp(p, x), test_params) for x in xgrid]
  @test test_grads == warp_grads # true 

  # Test 3: derivatives of the kernel with respect to warping parameters.
  test_grads   = SpectralKernels.kernel_warping_gradients(cfg, warp, xgrid, test_params;
                                                          backend=backend)
  kernel_grads = [ForwardDiff.gradient(p->kernel(x, p), test_params) for x in xgrid]
  @test maximum(norm.(test_grads .- kernel_grads)) < 1e-8

end

