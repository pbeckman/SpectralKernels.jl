
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
  xpairs = vec(collect(Iterators.product(eachindex(xgrid), eachindex(xgrid))))

  # for now, removing origin calls until the doubled lag 0 issue is sorted out.
  filter!(x->x[1]!=x[2], xpairs)

  model = SpectralModel(;cfg=cfg, warp=warp, sdf_param_indices=(1,),
                        warp_param_indices=(2,), singularity_param_index=0,
                        pts=xgrid, kernel_index_pairs=xpairs, verbose=false)

  function baseline_kernel_sum(params)
    sum(jk->kernel(xgrid[jk[1]], xgrid[jk[2]], params), xpairs)
  end

  function test_kernel_sum(params)
    gk = SpectralKernels.gen_kernel(model, params)
    sum(values(gk.store))
  end

  g_ref  = ForwardDiff.gradient(p->baseline_kernel_sum(p)*sqrt(p[1]*p[2]), test_params)
  g_test = ForwardDiff.gradient(p->test_kernel_sum(p)*sqrt(p[1]*p[2]), test_params)

  @test g_ref ≈ g_test

end

