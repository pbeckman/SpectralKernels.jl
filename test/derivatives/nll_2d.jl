
@testset "2D: nll + parameter indexing" begin

  TEST_PARAMS = [1.5, 0.1, 1.3, 0.8]

  iso_sdf(norm_w, range, nu) = (range^2 + norm_w^2)^(-nu - 1) # 2D!
  iso_kernel(t, par) = matern_cov(norm(t), vcat(1.0, par[3:4]); d=2)

  # this function only sees warping parameters, so indexing always starts from 1.
  warp(par, x) = SA[x[1]*par[1], x[1]*x[2]/par[2]]

  # this function 
  function warp_kernel(x, y, par) 
    warp_params = par[1:2]
    sdf_params  = par[3:4]
    t = warp(warp_params, x) - warp(warp_params, y)
    matern_cov(norm(t), vcat(1.0, sdf_params); d=2)
  end

  # Let's actually pick some points and work through a full example. Let's say
  # that we're on a regular lattice on [0,1]. 
  g1d   = range(0.1, 1.0, length=4)
  pts   = vec(SVector{2,Float64}.(Iterators.product(g1d, g1d)))
  model = SpectralModel(iso_sdf, pts; warp=warp,
                        warp_param_indices=1:2,
                        sdf_param_indices=3:4, tol=1e-12)

  function ref_nll(params)
    v  = ones(length(pts))
    M  = [warp_kernel(ptj, ptk, params) for ptj in pts, ptk in pts]
    Mf = cholesky!(Symmetric(M))
    (logdet(Mf) + sum(abs2, Mf.U'\v))/2
  end

  function test_nll(params)
    v   = ones(length(pts))
    ker = gen_kernel(model, params)
    M   = [ker(ptj, ptk) for ptj in pts, ptk in pts]
    M[1,1] = M[2,2] # TODO (cg 2026/02/10 17:56): wat
    Mf  = cholesky!(Symmetric(M))
    (logdet(Mf) + sum(abs2, Mf.U'\v))/2
  end

  @test isapprox(ForwardDiff.gradient(ref_nll, TEST_PARAMS),
                 ForwardDiff.gradient(test_nll, TEST_PARAMS);
                 atol=5e-5)

end

