
module SpectralKernelsForwardDiffExt

  using SpectralKernels, ForwardDiff
  using SpectralKernels.LinearAlgebra

  function SpectralKernels.gen_kernel(sm::SpectralModel, 
                                      params::Vector{ForwardDiff.Dual{T,Float64,N}}) where{T,N}
    params_primal = ForwardDiff.value.(params)
    out = SpectralKernels.gen_kernel(sm, params_primal)
    J   = SpectralKernels.gen_kernel_jacobian(sm, params_primal; 
                                              backend=SpectralKernels.AutoForwardDiff())
    _keys = collect(keys(out.store))
    vec_of_duals = map(enumerate(_keys)) do (j, key_j)
      partials = ntuple(k -> sum(J[j, m] * params[m].partials[k] for m = 1:length(params)), N)
      ForwardDiff.Dual{T}(out.store[key_j], partials)
    end
    SpectralKernels.SpectralKernel(Dict(zip(_keys, vec_of_duals)))
  end

end

