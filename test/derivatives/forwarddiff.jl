
@testset "Hooking into ForwardDiff.jl" begin

  # specify the model and the kernel, and then the warping function:
  iso_sdf(w, alpha)    = exp(-alpha*abs(w))
  iso_kernel(r, alpha) = 2*alpha/(alpha^2 + (2*pi*r)^2)
  warp(params, x)      = x^params[1] 

  kernel(x, y, params) = iso_kernel(abs(warp((params[2],), x) - warp((params[2],), y)), params[1])

  test_params = [1.1, 0.1]
  backend     = AutoForwardDiff()

  cfg    = SpectralKernels.AdaptiveKernelConfig(iso_sdf; tol=1e-12)
  xgrid  = collect(range(0.0, 1.0, length=20))

  model  = SpectralModel(iso_sdf, xgrid; warp, sdf_param_indices=1, 
                         warp_param_indices=2, verbose=false)

  function baseline_kernel_sum(params)
    sum(xy->kernel(xy[1], xy[2], params), Iterators.product(xgrid, xgrid))
  end

  function test_kernel_sum(params)
    gk = SpectralKernels.gen_kernel(model, params)
    sum(xy->gk(xy[1], xy[2], params), Iterators.product(xgrid, xgrid))
  end

  g_ref  = ForwardDiff.gradient(p->baseline_kernel_sum(p)*sqrt(p[1]*p[2]), test_params)
  g_test = ForwardDiff.gradient(p->test_kernel_sum(p)*sqrt(p[1]*p[2]), test_params)

  @test g_ref ≈ g_test

end

