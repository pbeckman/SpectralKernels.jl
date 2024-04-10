
struct SpectralKernel{T}
  store::Dict{Float64, T}
end

function SpectralKernel(config::AdaptiveKernelConfig{Dummy}, 
                        sdf::S,
                        xs::Vector{Float64}, 
                        parmsp::NTuple{N,Float64};
                        singularity=false,
                        nugget=false) where{S,N}
  (parms, singp, nugp) = @match (singularity, nugget) begin
    (false, false) => (parmsp, config.p, NaN)
    (true,  false) => (parmsp[1:(N-1)], parmsp[N], NaN)
    (false, true)  => (parmsp[1:(N-1)], config.p, parmsp[N])
    (true, true)   => (parmsp[1:(N-2)], parmsp[N-1], parmsp[N])
  end
  psdf  = ParametricFun(sdf, parms)
  pcfg  = AdaptiveKernelConfig(psdf; tol=config.tol, quadspec=config.quadspec, 
                               p=singp, logw=config.logw, tail=config.tail)
  kvals = kernel_values(pcfg, xs)[1]
  if nugget
    zix = findfirst(iszero, xs)
    kvals[zix] += nugp^2
  end
  SpectralKernel(Dict(zip(xs, kvals)))
end

function dSpectralKernel(config::AdaptiveKernelConfig{Dummy}, 
                         sdf::S,
                         xs::Vector{Float64}, 
                         parmsp::NTuple{N, ForwardDiff.Dual{T,Float64,N}};
                         singularity=false,
                         nugget=false) where{T,S,N}
  # compute the base values:
  pr_val = ForwardDiff.value.(parmsp)
  # now split the parameters based on the other bells and whistles in the config
  # (namely the singularity parameter and the nugget).
  (parms, singp, nugp, singix, gradN) = @match (singularity, nugget) begin
    (false, false) => (tuple(pr_val...), config.p, 0.0, 0, N)
    (true,  false) => (tuple(pr_val[1:(N-1)]...), pr_val[N], 0.0, N, N)
    (false, true)  => (tuple(pr_val[1:(N-1)]...), config.p, pr_val[N], 0, N-1)
    (true, true)   => (tuple(pr_val[1:(N-2)]...), pr_val[N-1], pr_val[N], N-1, N-1)
  end
  psdf = ParametricFun(sdf, parms)
  # IMPORTANT: the singularity term is actually pre-baked into the weights for
  # the quadrature rules. So you can't actually just pre-compute them once for
  # each SpectralModel object and re-use every time here. 
  pcfg  = AdaptiveKernelConfig(psdf; tol=config.tol, quadspec=config.quadspec, 
                               p=singp, logw=config.logw, tail=config.tail)
  kvals = kernel_values(pcfg, xs)[1]
  if nugget
    zix = findfirst(iszero, xs)
    kvals[zix] += nugp^2
  end
  # prepare the component derivatives of the SDF:
  singN = singularity ? Val(N-1) : Val(N)
  dsdfv = component_derivatives(sdf, singN) 
  # now loop over and compute those kvals:
  dvals = map(1:gradN) do j
    dcfgj = if j == singix
      AdaptiveKernelConfig(psdf; p=singp, quadspec=pcfg.quadspec, tol=pcfg.tol,
                           tail=pcfg.tail, logw=true)
    else
      AdaptiveKernelConfig(ParametricFun(dsdfv[j], parms); p=singp, quadspec=pcfg.quadspec, 
                           tol=pcfg.tol, tail=pcfg.tail, logw=false)
    end
    kernel_values(dcfgj, xs)[1]
  end
  if nugget
    zix     = findfirst(iszero, xs)
    kv      = zeros(length(xs))
    kv[zix] = 2*nugp
    push!(dvals, kv)
  end
  # now re-organize the output into the form that ForwardDiff expects:
  storevals = map(eachindex(xs)) do j
    ForwardDiff.Dual{T}(kvals[j], ForwardDiff.Partials(ntuple(k->dvals[k][j], N)))
  end
  SpectralKernel(Dict(zip(xs, storevals)))
end

function SpectralKernel(config::AdaptiveKernelConfig{Dummy}, 
                        sdf::S,
                        xs::Vector{Float64}, 
                        parmsp::NTuple{N, ForwardDiff.Dual{T,Float64,N}};
                        singularity=false,
                        nugget=false) where{T,S,N}
  dSpectralKernel(config, sdf, xs, parmsp; 
                  singularity=singularity, nugget=nugget)
end


function (sk::SpectralKernel)(x::Float64, y::Float64, dummy=nothing)
  xmy = abs(x-y)
  out = get(sk.store, xmy, NaN)
  isnan(out) && throw(error("Kernel requested at un-computed location: $x-$y = $xmy is not a pre-computed value."))
  out
end

function (sk::SpectralKernel)(x::SVector{1,Float64}, 
                              y::SVector{1,Float64}, 
                              dummy=nothing)
  sk(x[1], y[1], dummy)
end

