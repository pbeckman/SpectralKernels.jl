
iso_sdf(norm_w, ν) = matern_sdf(norm_w, (1.0, 1.0, ν); d=2)

warp(par, x) = SA[x[1]*par[1], x[1]*x[2]/par[2]]

function warp_kernel(x, y, params)
  warpx = warp(params[1:2], x)
  warpy = warp(params[1:2], y)
  sing_matern_cov(norm(warpx - warpy) + 1e-30, -params[end], [1.0, 1.0, params[3]]; d=2)
end

g1d   = range(0.1, 1.0, length=4)
pts   = vec(SVector{2,Float64}.(Iterators.product(g1d, g1d)))
model = SpectralModel(iso_sdf, pts; warp=warp,
                      warp_param_indices=1:2,
                      sdf_param_indices=3, 
                      singularity_param_index=4,
                      tol=1e-12)

function ref_nll(params)
  v  = ones(length(pts))
  M  = [warp_kernel(x, y, params) for x in pts, y in pts]
  Mf = cholesky!(Symmetric(M))
  (logdet(Mf) + sum(abs2, Mf.U'\v))/2
end

function test_nll(params)
  v   = ones(length(pts))
  ker = gen_kernel(model, params)
  M   = [ker(x, y) for x in pts, y in pts]
  Mf  = cholesky!(Symmetric(M))
  (logdet(Mf) + sum(abs2, Mf.U'\v))/2
end

ref_grad  = ForwardDiff.gradient(ref_nll,  [0.8, 4.1, 1.75, 0.4])
test_grad = ForwardDiff.gradient(test_nll, [0.8, 4.1, 1.75, 0.4])

@test isapprox(ref_grad, test_grad; rtol=1e-7)

