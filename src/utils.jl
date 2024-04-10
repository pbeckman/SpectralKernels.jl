
struct Dummy <: Function end

# make (v1, v2) .+= (w1, w2) -> (v1.+w1, v2.+w2) work without allocating
inplace_add_tuples!(T1, T2) = map(Ts -> Ts[1] .+= Ts[2], zip(T1, T2))

# make (v1, v2) .*= x -> (v1.*x, v2.*x) work without allocating
inplace_mul_tuple!(T, x) = map(t -> t .*= x, T)

finufft1d3(w::Vector, s::Vector, x::Vector) = vec(nufft1d3(w, s, +1, 1e-15, 2pi*x))

function print_panel_info(xs, true_is_converged, lowest_unconv_ix, 
                          highest_unconv_ix, a, b)
  @printf("\nintegrating panel w ∈ [%.2e, %.2e] (length %.2e) to resolve %i points x ∈ [%.2e, %.2e]\n", 
          a, b, b-a, sum(!, true_is_converged), xs[lowest_unconv_ix], xs[highest_unconv_ix])
  nothing
end

function print_panel_convergence(max_I_error, tol, _a, _b)
  if max_I_error > tol
      @printf("\tsubpanel w ∈ [%.2e, %.2e] did not converge to tolerance %.2e with max error %.2e\n", 
              _a, _b, tol, max_I_error)
  else
    @printf("\tsubpanel w ∈ [%.2e, %.2e] converged to tolerance %.2e with max error %.2e\n",
              _a, _b, tol, max_I_error)
  end
  nothing
end

function check_subdivide_failure(a, b, xs; verbose=false)
  abs(b - a) > 1e-16 && return
  if verbose
    println("\t(Sub-)panel integration failed:")
    println("\t\t-- (a,b): ($a, $b)")
    println("\t\t-- maximum frequency: $(maximum(xs))")
  end
  throw(error("The sub-interval (a, b) = ($a, $b) has been split too many times (b - a < 1e-16). Exiting to avoid infinite splitting."))
end

# TODO: pick this thoughtfully
nufft_quad_size_cutoff(n_no, n_x) = n_no*n_x > 2^18

function build_dense_cov_matrix(cfg, pts)
  npt = length(pts)
  xs_unsort = [
      0.0; 
      vcat([[abs(pts[i] - pts[j]) for j=i+1:npt] for i=1:npt-1]...)
      ]
  perm = sortperm(xs_unsort)
  xs   = xs_unsort[perm]
  n    = length(xs)

  M_vals        = Vector{Float64}(undef, n)
  M_vals[perm] .= kernel_values(cfg, xs, verbose=false)[1]

  M_fourier = Matrix{Float64}(undef, npt, npt)
  j = 1
  for i=1:npt
      M_fourier[i, i+1:end] .= M_vals[(j+1):(j+npt-i)]
      M_fourier[i+1:end, i] .= M_vals[(j+1):(j+npt-i)]
      M_fourier[i, i] = M_vals[1]
      j += npt-i
  end

  return M_fourier
end