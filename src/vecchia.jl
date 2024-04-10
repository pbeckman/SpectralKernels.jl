
struct SpectralModel{S}
  sdf::S
  kcfg::AdaptiveKernelConfig{Dummy}
  vcfg::Vecchia.VecchiaConfig{Float64,1,Dummy}
  xs::Vector{Float64}
  singularity::Bool
  nugget::Bool
end

# just for 1D data. Max lazy for now.
function compute_interpoint_distances(cfg)
  out = Float64[]
  sizehint!(out, sum(length, cfg.pts)*sum(length, cfg.condix))
  for j in eachindex(cfg.pts)
    pts = cfg.pts[j]
    # marginal variance values/matrices for leaves:
    for (ptj, ptk) in Iterators.product(pts, pts)
      push!(out, norm(ptj - ptk))
    end
    # cross covariances between leaves and conditioning sets.
    cixj = cfg.condix[j]
    for (j,k) in Iterators.product(cixj, cixj)
      (condj, condk) = (cfg.pts[j], cfg.pts[k])
      # to avoid double counting, we add the cross between the marginal leaf and
      # conditioning set when one of the indices is one.
      if k == cixj[1]
        for (ptj, ptk) in Iterators.product(pts, condj)
          push!(out, norm(ptj - ptk))
        end
      end
      # now add the cond-cross-cond interpoint distances.
      for (ptj, ptk) in Iterators.product(condj, condk)
        push!(out, norm(ptj - ptk))
      end
    end
  end
  sort!(out)
  unique!(out)
  out
end

function SpectralModel(sdf, cfg; singularity=false, nugget=false, kwargs...)
  if singularity || nugget
    @info "At present, singularity and nugget parameters are parsed as the last one or two entries in the parameter vector. If you have both a singularity AND a nugget, the nugget goes in the last position." maxlog=1
  end
  kcfg = AdaptiveKernelConfig(Dummy(); kwargs...)
  vcfg = Vecchia.VecchiaConfig(Dummy(), cfg.data, cfg.pts, cfg.condix)
  xs = compute_interpoint_distances(cfg)
  SpectralModel(sdf, kcfg, vcfg, xs, singularity, nugget)
end

function _nll(sm::SpectralModel{S}, params) where{S}
  (kcfg, vcfg) = (sm.kcfg, sm.vcfg)
  sk   = SpectralKernel(kcfg, sm.sdf, sm.xs, tuple(params...); 
                        singularity=sm.singularity, nugget=sm.nugget)
  vcfg = Vecchia.VecchiaConfig(sk, vcfg.data, vcfg.pts, vcfg.condix)
  Vecchia.nll(vcfg, params)
end

function _nll(sm::SpectralModel{S}, params::AbstractVector{ForwardDiff.Dual{T,Float64,N}}) where{S,T,N}
  (kcfg, vcfg) = (sm.kcfg, sm.vcfg)
  dsk  = dSpectralKernel(kcfg, sm.sdf, sm.xs, tuple(params...), 
                         singularity=sm.singularity, nugget=sm.nugget)
  vcfg = Vecchia.VecchiaConfig(dsk, vcfg.data, vcfg.pts, vcfg.condix)
  Vecchia.nll(vcfg, params)
end

