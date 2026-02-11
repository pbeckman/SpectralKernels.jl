
module SpectralKernelsForwardDiffExt

  using SpectralKernels, ForwardDiff
  using SpectralKernels.LinearAlgebra

  function SpectralKernels.gen_kernel(sm::SpectralModel, 
                                      params::Vector{ForwardDiff.Dual{T,Float64,N}}) where{T,N}
    params_primal = ForwardDiff.value.(params)
    out = SpectralKernels.gen_kernel(sm, params_primal)
    J   = SpectralKernels.gen_kernel_jacobian(sm, params_primal; 
                                              backend=SpectralKernels.AutoForwardDiff())
    dual_pairs = map(enumerate(sm.kernel_index_pairs)) do (j, ik)
      (ptj, ptk) = (sm.pts[ik[1]], sm.pts[ik[2]])
      partials   = ntuple(k -> sum(J[j, m] * params[m].partials[k] for m = 1:length(params)), N)
      (ptj, ptk) => ForwardDiff.Dual{T}(out.store[(ptj, ptk)], partials)
    end
    SpectralKernels.SpectralKernel(Dict(dual_pairs))
  end

end

