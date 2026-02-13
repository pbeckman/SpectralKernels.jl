
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
  xpairs = vec(collect(Iterators.product(eachindex(xgrid), eachindex(xgrid))))

  model = SpectralModel(;cfg=cfg, warp=warp, sdf_param_indices=(1,),
                        warp_param_indices=(2,), singularity_param_index=0,
                        pts=xgrid, kernel_index_pairs=xpairs, verbose=false)

  J_test = SpectralKernels.gen_kernel_jacobian(model, test_params; backend=backend)

  J_ref = ForwardDiff.jacobian(params -> begin
                                 [kernel(xgrid[jk[1]], xgrid[jk[2]], params) for jk in xpairs]
                               end,
                               test_params)

  @test maximum(abs, J_test - J_ref) < 1e-8

end

