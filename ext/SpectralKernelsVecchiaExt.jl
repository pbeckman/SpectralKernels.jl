
module SpectralKernelsVecchiaExt

  using SpectralKernels, Vecchia, ForwardDiff
  using SpectralKernels.LinearAlgebra
  import Vecchia: SingletonVecchiaApproximation, ZeroMean
  import SpectralKernels: SpectralLikelihood

  function SpectralKernels.SpectralLikelihood(::Type{<:VecchiaApproximation},
                                              model::SpectralModel, data; kwargs...)
    dummy_cfg = VecchiaApproximation(model.pts, nothing, data; kwargs...)
    kernel_index_pairs = Vecchia.tile_pairs(dummy_cfg.condix)
    new_model = SpectralModel(model.cfg, model.warp, model.sdf_param_indices,
                              model.warp_param_indices, model.singularity_param_index,
                              dummy_cfg.pts, kernel_index_pairs, model.verbose)
    SpectralLikelihood(new_model, dummy_cfg)
  end
                                
  function (sl::SpectralLikelihood{S,<:SingletonVecchiaApproximation})(params) where{S}
    kernel   = gen_kernel(sl.model, params)  
    base_cfg = sl.spec
    appx     = SingletonVecchiaApproximation(ZeroMean(), kernel, 
                                             base_cfg.data, base_cfg.pts, 
                                             base_cfg.condix, base_cfg.perm)
    appx(params)
  end

  function SpectralKernels.simulate(sl::SpectralLikelihood{S,<:SingletonVecchiaApproximation}, 
                                    params; z=randn(length(sl.spec.pts))) where{S}
    kernel   = gen_kernel(sl.model, params)  
    base_cfg = sl.spec
    appx     = SingletonVecchiaApproximation(ZeroMean(), kernel, 
                                             base_cfg.data, base_cfg.pts, 
                                             base_cfg.condix, base_cfg.perm)
    rchol(appx, params).U'\z
  end

  function _nll_grad_fish(sl::SpectralLikelihood{S,C}, 
                          params::Vector{Float64}) where{S,C<:SingletonVecchiaApproximation}
    tag = ForwardDiff.Tag(SpectralLikelihood{S,C}, Float64)
    N   = length(params)
    duals = map(1:length(params)) do j
      pj = ForwardDiff.Partials(ntuple(k->Float64(k==j), N))
      ForwardDiff.Dual{typeof(tag)}(params[j], pj)
    end
    kernel   = gen_kernel(sl.model, duals)
    base_cfg = sl.spec
    appx     = SingletonVecchiaApproximation(ZeroMean(), kernel, 
                                             base_cfg.data, base_cfg.pts, 
                                             base_cfg.condix, base_cfg.perm)
    Vecchia._nll_grad_fish(appx, duals)
  end

  function Vecchia._hessian(cw::Vecchia.CachingForwardADWrapper{SpectralLikelihood{S,C},G,H,R}, 
                            params::Vector{Float64}) where{S,C,G,H,R}
    @info "Using specialized SpectralLikelihood e-fish routine..." maxlog=1
    if haskey(cw.cache, params)
      params_result = cw.cache[params]
      isnothing(params_result.hessian) || return params_result.hessian
    end
    result = _nll_grad_fish(cw.fn, params)
    store  = Vecchia.EvaluationResult(result[1], result[2], Symmetric(result[3]))
    cw.cache[params] = store
    result[end]
  end

end

