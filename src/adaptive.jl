
Base.@kwdef struct AdaptiveKernelConfig
  dim::Int64                    # dimension of isotropic sdf
  alpha::Float64                # origin power singularity |w|^{-α} * sdf(w)
  tol::Float64                  # relative tolerance
  convergence_criteria::Symbol  # :tails, :panel, or :both (default)
  tail::Union{Float64, Nothing} # power law exponent for tail decay (optional)
  logw::Bool                    # whether to include log(w) singularity
  quadspec::Tuple{Int64, Int64} # (m, k): k many m-node panels per NUFFT
  legrule::QuadRule             # Legendre quadrature rule
  jacrule::QuadRule             # Jacobi quadrature rule
  buffers::Tuple{               # m- and 2m-node locations and integrands
    Vector{Float64}, Vector{ComplexF64},
    Vector{Float64}, Vector{ComplexF64} 
    } 
  splittingheap::SplittingHeap  # heap of subintervals to be integrated
end

struct Integrand{S,dS}
  f::S
  df::dS
end

function AdaptiveKernelConfig(sdf::S; 
  dim=1, alpha=0.0, tol::Float64=1e-8, convergence_criteria::Symbol=:both, tail=nothing, logw=false, 
  quadspec::Tuple{Int64, Int64}=(2^12, 2^4)) where{S}

  if !(convergence_criteria in [:panel, :tails, :both]) 
    error("Argument convergence_criteria must be one of :panel, :tails, :both.")
  end

  if tol < 1e-12 && prod(quadspec) > 2^12
    @warn("Tolerances ε < 1e-12 are not recommended. Switching to a smaller quadrature rule for higher accuracy (but slower) computations.")
    quadspec = (2^12, 1)
  end

  q = (dim == 1) ? 0 : 1
  m, k = quadspec                     
  legrule = QuadRule(m; case=:legendre)
  jacrule = (q-alpha != 0.0) ? QuadRule(m; case=:jacobi, p=q-alpha) : legrule
  buffers = (
    Vector{Float64}(undef,  m*k), Vector{ComplexF64}(undef,  m*k), 
    Vector{Float64}(undef, 2m*k), Vector{ComplexF64}(undef, 2m*k)
    )
  AdaptiveKernelConfig(
    dim, alpha, tol, convergence_criteria, tail, logw, 
    quadspec, legrule, jacrule, buffers, SplittingHeap())
end

quadsz(ac::AdaptiveKernelConfig) = prod(ac.quadspec)

function kernel_values(integrand::Integrand{S,dS},
                       xs::AbstractVector{Float64},
                       config::AdaptiveKernelConfig; verbose=false) where{S,dS}
  issorted(xs) || throw(error("locations must be provided in sorted order."))
  # allocate some full-length buffers to re-use throughout the main loop:
  (ks, errs) = (zeros(Float64, length(xs)) for _ in 1:2)
  highest_unconv_ix = length(xs)

  # get the integrand and its derivative:
  (fun, dfun) = (integrand.f, integrand.df)

  # initialize relavant constants and variables
  quadm     = quadsz(config)
  conv_crit = config.convergence_criteria
  (a, b) = (0.0, 0.0)
  (c, d) = (NaN, NaN)
  # multiplicative constant m and exponent q so that m ∫ w^{q-α} f(w) ϕ(2πrw) dw
  # gives the correct Fourier integral where ϕ is cos in 1D or J_0 in 2D
  m, q = (config.dim == 1) ? (2, 0) : (2pi, 1)

  # compute K(0) using QuadGK
  @timeit TIMER "K(0)" begin
    L = 1.0
    while L^(q-config.alpha) * abs(fun(L)) > abs(fun(0))/2
      L *= 2
    end
    f(w) = (w*L)^(q-config.alpha) * (config.logw ? log(w*L) : 1) * fun(w*L) * L
    k0, k0_err = m .* quadgk(
      w -> f(w), 0, Inf, 
      atol=0.0, rtol=min(1e-8, 1e-2*config.tol)
      )
  end
  if iszero(xs[1])
    ks[1], errs[1] = (k0, k0_err)
  end

  # main loop:
  while highest_unconv_ix > 0 && xs[highest_unconv_ix] > 0
    # choose panel so that b-a is the distance in w-space for which m points
    # acheive the Nyquist sampling rate for exp(2πiwx) for all unconverged x
    (a, b) = (b, b + quadm / (2*xs[highest_unconv_ix]))
    # print output about panel:
    verbose && print_panel_info(xs, highest_unconv_ix, a, b)
    # add kernel values and quadrature errors from this panel to buffers
    @timeit TIMER "interval integral" begin
      (panel_ks, panel_errs) = fourier_integrate_interval(
        a, b, config,
        xs[1:highest_unconv_ix], abs(k0), verbose
        )
    end
    # add integral of new panel and panel quadrature error
    @timeit TIMER "add panels to integral" begin
      view(ks,   1:highest_unconv_ix) .+= panel_ks
      view(errs, 1:highest_unconv_ix) .+= panel_errs
    end
    # estimate tail power law prefactor and decay rate
    @timeit TIMER "estimate tail decay" begin
      (c, d) = (conv_crit == :panel) ? (NaN, NaN) : estimate_tail_decay(integrand, a, b, d=config.tail)
    end
    if (isnan(c) || isnan(d)) && (conv_crit != :panel)
      # if tail gives NaNs, it's probably because it decayed too fast, e.g. was
      # exponential, for which truncation error is small, and we can rely on
      # panel contribution to control error
      verbose && @printf("\talgebraic tail estimate failed -- using convergence_criteria = :panel\n")
      conv_crit = :panel
    else
      if conv_crit != :panel
        verbose && @printf("\talgebraic tail estimate S(w) ≈ %.2e * w^(%.2f)\n", c, d)
      end
    end
    # check convergence of xs in decreasing order
    @timeit TIMER "check convergence" begin
      conv = true
      ix   = highest_unconv_ix
      while conv && ix > 0
        trunc_err = (conv_crit == :panel) ? 0 : truncation_error_estimate(
          b, xs[ix], c, d, config.dim
          )
        conv = check_convergence(
          trunc_err, panel_ks[ix], config.tol*abs(k0)/2, criteria=conv_crit
          )
        if conv
          errs[ix] += 2*trunc_err
          ix -= 1
        end
      end
      highest_unconv_ix = ix
    end
  end
  return (ks, errs)
end

function estimate_tail_decay(integrand::Integrand, a, b; d=nothing)
  # get the integrand and its derivative:
  (fun, dfun) = (integrand.f, integrand.df)
  # number of points to fit on panel
  nf = 1000
  # choose some frequency points on the last panel to extrapolate from
  ws = range(a + (b-a), stop=b, length=nf)
  if isnothing(d)
    # linear least squares in log space to estimate d
    tmp = @. log(abs(fun(ws)))
    logc, d = [ones(nf) log.(ws)] \ tmp
  end
  d += -config.alpha
  # compute the least squares estimate for c
  c = sum(ws.^(d - config.alpha) .* abs.(fun.(ws))) / sum(ws.^(2d))
  (c, d)
end

function truncation_error_estimate(b, x, c, d, dim)
    # compute analytic truncation error of power law Fourier integral from b to
    # Inf with multiplicative constant c and power law exponent d
    return min(
      -c/(d+dim)*b^(d+dim), 
      c*b^(d+(dim-1)/2) / (2pi*x^((dim+1)/2))
      )
end

function check_convergence(trunc_err, panel_k, tol; criteria=:both)
  (criteria == :panel || trunc_err < tol) && (criteria == :tails || abs(panel_k) < tol)
end
