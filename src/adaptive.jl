
Base.@kwdef struct AdaptiveKernelConfig{S}
  sdf::S                        # spectral density function
  p::Float64                    # origin power singularity |w|^p * sdf(w)
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

function AdaptiveKernelConfig(sdf::S; 
  tol::Float64=1e-10, convergence_criteria=:both, p=0.0, tail=nothing, logw=false, 
  quadspec::Tuple{Int64, Int64}=(2^12, 2^4)) where{S}

  if !(convergence_criteria in [:panel, :tails, :both]) 
    error("Argument convergence_criteria must be one of :panel, :tails, :both.")
  end

  if tol < 1e-12 && prod(quadspec) > 2^12
    @warn("Tolerances ε < 1e-12 are not recommended. Switching to a smaller quadrature rule for higher accuracy (but slower) computations.")
    quadspec = (2^12, 1)
  end

  m, k = quadspec                     
  legrule = QuadRule(m; case=:legendre)
  jacrule = (p != 0.0) ? QuadRule(m; case=:jacobi, p=p) : legrule
  buffers = (
    Vector{Float64}(undef,  m*k), Vector{ComplexF64}(undef,  m*k), 
    Vector{Float64}(undef, 2m*k), Vector{ComplexF64}(undef, 2m*k)
    )
  AdaptiveKernelConfig{S}(
    sdf, p, tol, convergence_criteria, tail, logw, 
    quadspec, legrule, jacrule, buffers, SplittingHeap())
end

quadsz(ac::AdaptiveKernelConfig) = prod(ac.quadspec)

function kernel_values(config::AdaptiveKernelConfig{S}, 
                       xs::Vector{Float64}; verbose=false, save_quad=false) where{S}
  issorted(xs) || throw(error("locations must be provided in sorted order."))
  # allocate some full-length buffers to re-use throughout the main loop:
  (ks, errs) = (zeros(length(xs)) for _ in 1:2)
  (true_is_converged, false_is_converged) = (fill(false, length(xs)) for _ in 1:2)

  # initialize relavant constants and variables
  quadm     = quadsz(config)
  conv_crit = config.convergence_criteria
  (a, b) = (0.0, 0.0)
  (c, d) = (NaN, NaN)

  # compute K(0) using QuadGK
  @timeit TIMER "K(0)" begin
    L = 1.0
    while L^config.p * abs(config.sdf(L)) > abs(config.sdf(0))/2
      L *= 2
    end
    f(w) = (w*L)^config.p * (config.logw ? log(w*L) : 1) * config.sdf(w*L) * L
    k0, k0_err = 2 .* quadgk(
      w -> f(w), 0, Inf, 
      atol=0.0, rtol=min(1e-8, 1e-2*config.tol)
      )
  end
  if iszero(xs[1])
    ks[1], errs[1] = (k0, k0_err)
    true_is_converged[1]  = true
  end

  # main loop:
  while !all(true_is_converged)
    # update the "false_is_converged" indices:
    broadcast!(!, false_is_converged, true_is_converged)
    lowest_unconv_ix  = findfirst(false_is_converged)
    highest_unconv_ix = findlast(false_is_converged)
    # choose panel so that b-a is the distance in w-space for which 
    # m points acheive the Nyquist sampling rate for exp(2πiwx) for all 
    # unconverged x
    (a, b) = (b, b + quadm / (2*xs[highest_unconv_ix]))
    # print output about panel:
    verbose && print_panel_info(xs, true_is_converged, lowest_unconv_ix, 
                                highest_unconv_ix, a, b)
    # add kernel values and quadrature errors from this panel to buffers
    xs_unconv = xs[false_is_converged]
    @timeit TIMER "interval integral" begin
      (panel_ks, panel_errs) = 2 .* fourier_integrate_interval(
        a, b, config,
        xs_unconv, abs(k0), verbose, save_quad=save_quad
        )
    end
    # add integral of new panel and panel quadrature error
    @timeit TIMER "bitarray view operations" begin
      @timeit TIMER "add panels to integral" begin
        view(ks,   false_is_converged) .+= panel_ks
        view(errs, false_is_converged) .+= panel_errs
      end
    end
    # estimate tail power law prefactor and decay rate
    @timeit TIMER "estimate tail decay" begin
      (c, d) = (conv_crit == :panel) ? (NaN, NaN) : estimate_tail_decay(config, a, b, d=config.tail)
    end
    if (isnan(c) || isnan(d)) && (conv_crit != :panel)
      # if tail gives NaNs, it's probably because it decayed too fast,
      # e.g. was exponential, for which truncation error is small, and we can
      # rely on panel contribution to control error
      verbose && @printf("\talgebraic tail estimate failed -- using convergence_criteria = :panel\n")
      conv_crit = :panel
    else
      if conv_crit != :panel
        verbose && @printf("\talgebraic tail estimate S(w) ≈ %.2e * w^(%.2f)\n", c, d)
      end
    end
    # check if any xs are still unconverged
    @timeit TIMER "bitarray view operations" begin
      @timeit TIMER "update converence bitarray" begin
        for (j, ix) in enumerate(view(1:length(xs), false_is_converged))
          trunc_err = truncation_error_estimate(
            b, xs[ix], c, d
            )
          conv = check_convergence(
            trunc_err, panel_ks[j], config.tol*abs(k0)/2, criteria=conv_crit
            )
          if conv
            errs[ix] += 2*trunc_err
            true_is_converged[ix] = true
          end
        end
      end
    end
  end
  return (ks, errs)
end

function estimate_tail_decay(config, a, b; d=nothing)
  # number of points to fit on panel
  nf = 1000
  # choose some frequency points on the last panel to extrapolate from
  ws = range(a + (b-a), stop=b, length=nf)
  if isnothing(d)
    # linear least squares in log space to estimate d
    tmp = log.(abs.(config.sdf.(ws)))
    logc, d = [ones(nf) log.(ws)] \ tmp
  end
  d += config.p
  # compute the least squares estimate for c
  c = sum(ws.^(d + config.p) .* abs.(config.sdf.(ws))) / sum(ws.^(2d))
  (c, d)
end

function truncation_error_estimate(b, x, c, d)
    # compute analytic truncation error of power law Fourier integral 
    # from b to Inf with multiplicative constant c and power law exponent d
    # return -c*b^(d+1) * real(expint(-d, 2pi*im*b*x))
    return min(-c/(d+1)*b^(d+1), c*b^d/(2pi*x))
end

function check_convergence(trunc_err, panel_k, tol; criteria=:both)
  (criteria == :panel || trunc_err < tol) && (criteria == :tails || abs(panel_k) < tol)
end
