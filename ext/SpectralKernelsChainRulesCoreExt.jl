
module SpectralKernelsForwardDiffExt

  using SpectralKernels, ForwardDiff
  import SpectralKernels: gen_kernel, gen_kernel_jacobian

  # Not presently in use---I thought that DifferentiationInterface.jl somehow made
  # ForwardDiff ChainRules compatible...but clearly not.
  function ChainRulesCore.frule((Δself, Δsm, Δparams),
                                ::typeof(gen_kernel),
                                sm, params)
    out    = gen_kernel(sm, params)
    J      = gen_kernel_jacobian(sm, params; backend=AutoForwardDiff())
    Δval   = J*unthunk(Δparams)
    Δstore = Dict(zip(keys(out.store), Δval))
    Δout   = Tangent{SpectralKernel}(;store=Δstore)
    (out, Δout)
  end

  # UNTESTED
  function ChainRulesCore.rrule(::typeof(gen_kernel), sm, params)
    out    = gen_kernel(sm, params)
    J      = gen_kernel_jacobian(sm, params; backend=AutoForwardDiff())
    function pullback(yb)
      yb = unthunk(yb)
      yb_store = yb.store
      yb_vals  = collect(values(yb_store))
      dparams  = J'*yb_vals
      (NoTangent(), NoTangent(), dparams)
    end
    out, pullback
  end

end

