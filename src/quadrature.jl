
struct SplittingHeap
  segments::Vector{Tuple{Float64, Float64, Float64}}  
  len::Vector{Int64}
end

function SplittingHeap(;sz::Int64=5000)
  SplittingHeap(Vector{Tuple{Float64, Float64, Float64}}(undef, sz), [0])
end

function Base.push!(sh::SplittingHeap, xv)
  sh.len[1] += 1
  sh.segments[sh.len[]] = xv
  nothing
end

function Base.pop!(sh::SplittingHeap)
  iszero(sh.len[]) && throw(error("Can't pop from empty heap!"))
  out = sh.segments[sh.len[]]
  sh.len[1] -= 1
  out
end

Base.isempty(sh::SplittingHeap) = iszero(sh.len[])
drain(sh::SplittingHeap) = (sh.len[1] = 0)

struct QuadRule
  # static containers of the un-shifted nodes and weights:
  no1::Vector{Float64} 
  wt1::Vector{Float64} 
  no2::Vector{Float64} 
  wt2::Vector{Float64}
end

function QuadRule(m; case, p=0.0)
  if case == :legendre || iszero(p)
    (no1, wt1) = gausslegendre(m)
    (no2, wt2) = gausslegendre(2m)
  elseif case == :jacobi
    p <= -1.0 && throw(error("p needs to be in (-1.0, Inf) to be integrable"))
    (no1, wt1) = gaussjacobi(m,  0.0, p)
    (no2, wt2) = gaussjacobi(2m, 0.0, p)
  else
    throw(error("Options are case=:legendre or case=:jacobi"))
  end
  QuadRule(no1, wt1, no2, wt2)
end

function updatequadbufs!(buffers, legrule::QuadRule, jacrule::QuadRule, f::F, a, b; p=0) where{F}
  (no1, buf1, no2, buf2) = buffers
  (nol1, wtl1) = (legrule.no1, legrule.wt1)
  (nol2, wtl2) = (legrule.no2, legrule.wt2)
  m = length(nol1)
  k = div(length(buf1), m)

  subivlrange = range(a, b, length=k+1)
  subivliter  = enumerate(zip(subivlrange[1:end-1], subivlrange[2:end]))
  
  # if we're at the origin integrating a singularity, use Jacobi rule
  # rescale the quadrature weight by bmad2^p, but don't put w^p in the integrand
  if p != 0 && a == 0.0
    _, (_sa, _sb) = first(subivliter)
    (bmad2, bpad2) = ((_sb - _sa)/2, (_sb + _sa)/2)

    (noj1, wtj1) = (jacrule.no1, jacrule.wt1)
    (noj2, wtj2) = (jacrule.no2, jacrule.wt2)
    for j=1:m
      no1[j]  = bmad2*noj1[j] + bpad2
      buf1[j] = wtj1[j]*bmad2^(p+1) * f(no1[j])
    end
    for j=1:2m
      no2[j]  = bmad2*noj2[j] + bpad2
      buf2[j] = wtj2[j]*bmad2^(p+1) * f(no2[j])
    end
    
    # omit the first subpanel in the next loop
    subivliter = Iterators.drop(subivliter, 1)
  end

  # for the rest of the subpanels, use the Legendre rule
  # use the usual quadrature weights, but put w^p in the integrand
  for (i, (_sa, _sb)) in subivliter
    (bmad2, bpad2) = ((_sb - _sa)/2, (_sb + _sa)/2)
    for j=1:m
      no1[(i-1)*m+j]  = bmad2*nol1[j] + bpad2
      buf1[(i-1)*m+j] = wtl1[j]*bmad2 * no1[(i-1)*m+j]^p * f(no1[(i-1)*m+j]) 
    end
    for j=1:2m
      no2[(i-1)*2m+j]  = bmad2*nol2[j] + bpad2
      buf2[(i-1)*2m+j] = wtl2[j]*bmad2 * no2[(i-1)*2m+j]^p * f(no2[(i-1)*2m+j]) 
    end
  end

  (no1, buf1, no2, buf2)
end

function fourier_integrate_panel(buffers, legrule::QuadRule, jacrule::QuadRule, f, a, b, xs; p=0, dim=1)
  @show p
  check_subdivide_failure(a, b, xs)
  @timeit TIMER "update quadrature buffers" begin
    (no1, buf1, no2, buf2) = updatequadbufs!(
      buffers, legrule, jacrule, 
      f, a, b, p=p
      )
  end
  if nufft_quad_size_cutoff(length(no2), length(xs)) && length(xs) > 1
    if dim == 1
      @timeit TIMER "FINUFFT call" begin
        int1 = finufft1d3(no1, buf1, xs)
        int2 = finufft1d3(no2, buf2, xs)
      end
    else
      @timeit TIMER "NUFHT call" begin
        int1 = nufht(0, no1, buf1, 2pi*xs; tol=1e-15)
        int2 = nufht(0, no2, buf2, 2pi*xs; tol=1e-15)
      end
    end
  else
    if dim == 1
      @timeit TIMER "direct Fourier summation" begin
        xl   = length(xs)
        int1 = zeros(ComplexF64, xl)
        int2 = zeros(ComplexF64, xl)
        for j in eachindex(xs)
          @inbounds begin
            xj = xs[j]
            @simd for k in eachindex(no1)
              int1[j] += buf1[k]*cispi(2*no1[k]*xj)
            end
            @simd for k in eachindex(no2)
              int2[j] += buf2[k]*cispi(2*no2[k]*xj)
            end
          end
        end
      end
    else
      @timeit TIMER "direct Bessel summation" begin
        xl   = length(xs)
        int1 = zeros(Float64, xl)
        int2 = zeros(Float64, xl)
        for j in eachindex(xs)
          @inbounds begin
            xj = xs[j]
            @simd for k in eachindex(no1)
              int1[j] += buf1[k]*besselj0(2pi*no1[k]*xj)
            end
            @simd for k in eachindex(no2)
              int2[j] += buf2[k]*besselj0(2pi*no2[k]*xj)
            end
          end
        end
      end
    end
  end
  any(isnan, int1) || any(isnan, int2) && throw(error("NaN detected in panel integral..."))
  (int1, int2)
end

function fourier_integrate_interval(a, b, integrand, config, xs, k0, verbose)
  m, q = (config.dim == 1) ? (2, 0) : (2pi, 1)
  intervalheap = config.splittingheap
  push!(intervalheap, (a, b, config.tol))
  I   = zeros(length(xs))
  err = zeros(length(xs))
  (fun, dfun) = (integrand.f, integrand.df)
  while !isempty(intervalheap)
    # take a (sub-)interval:
    (_a, _b, _tol) = pop!(intervalheap)
    # now pick a rule and integrand based on p and where (_a, _b) is:
    if _a == 0.0 && q-config.alpha != 0.0
      # TODO (pb 1/16/26) : implement log singularity for dim = 2
      if config.logw
        # use integration by parts to evaluate the Fourier integral with an
        # additional log(w) singularity
        I0 = (_b)^(-config.alpha+1)*log(_b)*fun(_b)*cos.(2pi*_b*xs)
        @timeit TIMER "panel integral" begin
          (I1a, I2a) = fourier_integrate_panel(
            config.buffers, config.legrule, config.jacrule,
            w -> fun(w) + w*log(w)*dfun(w),
            _a, _b, xs, dim=config.dim, p=-config.alpha
            )
          (I1b, I2b) = fourier_integrate_panel(
            config.buffers, config.legrule, config.jacrule,
            w -> w*log(w)*fun(w),
            _a, _b, xs, dim=config.dim, p=-config.alpha
            )
        end
        I1 = (I0 .- real(I1a) .+ 2pi*xs .* imag.(I1b)) / (-config.alpha+1)
        I2 = (I0 .- real(I2a) .+ 2pi*xs .* imag.(I2b)) / (-config.alpha+1)
      else
        @show q, config.alpha, "Jacobi"
        @timeit TIMER "panel integral" begin
          (I1, I2) = fourier_integrate_panel(
            config.buffers, config.legrule, config.jacrule,
            w -> fun(w),
            _a, _b, xs, dim=config.dim, p=q-config.alpha
            )
        end
      end
    else
      @show q, config.alpha, "Legendre"
      @timeit TIMER "panel integral" begin
        (I1, I2) = fourier_integrate_panel(
          config.buffers, config.legrule, config.jacrule,
          w -> w^(q-config.alpha) * (config.logw ? log(w) : 1) * fun(w), 
          _a, _b, xs, dim=config.dim
          )
      end
    end
    I1 .= real.(I1)
    I2 .= real.(I2)
    # now analyze the error and decide if you need to split more:
    _err = abs.(I2 - I1)
    max_I_error = maximum(_err)
    verbose && print_panel_convergence(max_I_error, _tol, _a, _b)
    if max_I_error < config.tol*k0
      I   .+= I2
      err .+= _err
    else
      # give the first subpanel higher tolerance while still summing to _tol.
      # for all other panels, give each subpanel equal parts summing to _tol.
      # this avoids unnecessarily restrictive tolerances at the origin while 
      # still controlling the total absolute error.
      _tol_l, _tol_r = (_a == 0) ? (9_tol/10, _tol/10) : (_tol/2, _tol/2)
      push!(intervalheap, (_a, (_a + _b)/2, _tol_l))
      push!(intervalheap, ((_a + _b)/2, _b, _tol_r))
    end
  end
  drain(intervalheap)
  return m .* (I, err)
end

