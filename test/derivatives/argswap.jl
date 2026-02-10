
@testset "argswap/ParametricFunction/ParametricDerivative" begin

  fn(w, a, b, c) = a^2 + b*w^c

  dfn_da(w, a, b, c) = ForwardDiff.derivative(_a->fn(w, _a, b, c), a)
  dfn_db(w, a, b, c) = ForwardDiff.derivative(_b->fn(w, a, _b, c), b)
  dfn_dc(w, a, b, c) = ForwardDiff.derivative(_c->fn(w, a, b, _c), c)

  for abc in ((1.1, 2.1, 3.1), (3.1, 2.1, 1.1))
    # make the parametric function object:
    pfn = SpectralKernels.ParametricFunction(fn, abc)
    # make the derivative objects:
    d_da = SpectralKernels.ParametricDerivative(pfn, Val(1), AutoForwardDiff())
    d_db = SpectralKernels.ParametricDerivative(pfn, Val(2), AutoForwardDiff())
    d_dc = SpectralKernels.ParametricDerivative(pfn, Val(3), AutoForwardDiff())
    for w in (0.01, 0.1, 1.1, 11.1, 111.1)
      @test fn(w, abc...) == pfn(w)
      @test dfn_da(w, abc[1], abc[2], abc[3]) == d_da(w)
      @test dfn_db(w, abc[1], abc[2], abc[3]) == d_db(w)
      @test dfn_dc(w, abc[1], abc[2], abc[3]) == d_dc(w)
    end
  end

end

