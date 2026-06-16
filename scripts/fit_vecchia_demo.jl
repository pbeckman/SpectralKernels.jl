
using StaticArrays, SpectralKernels, Vecchia, ForwardDiff

# Hard-coded 1D SDF for a Matern-like model where you're allowed to have a
# rougher kernel.
function sdf(w, scale, alpha, nu)
  desired_rate = -2*(nu - 1/2)
  scale*(1 + abs(w)^alpha)^(desired_rate/alpha)
end

# Let's also write up a warping function, although in this case it will just
# effectively handle the range parameter (this speeds up the code a bit instead
# of putting it in the SDF function above because the derivatives are faster to
# compute this way---see the manuscript for details).
warp(params, x) = x/params[1]

# Let's just pick a few thousand points.
pts = SVector{1,Float64}.(sort(rand(5_000)))

# Now we make a SpectralModel. We'll have to decide how the parameters are
# ordered and tell the constructor which parameter indices are used for the
# warping function or the SDF function, because that impacts how derivatives for
# each are computed. In this case, let's say that our parameters are provided
# as:
#
# [scale_param, range_param, smoothness, alpha].
#
# Then the constructor looks like this:
smodel = SpectralModel(sdf, pts; warp=warp,
                       sdf_param_indices=[1,4,3],  # note the re-mapping to the order of sdf
                       warp_param_indices=2,       # you are welcome to provide just a scalar
                       kernel_index_pairs=[(0,0)], # a dummy value that will be updated below
                       verbose=true, # pass in any kwargs you would for AdaptiveKernelConfig!
                       tol=1e-12) # tolerance for the integration

# Now we specify a SpectralLikelihood with a Vecchia-based likelihood
# approximation as below, where keyword arguments are passed to the Vecchia
# constructor.
data         = ones(length(pts)) # a simple placeholder
spectral_nll = SpectralLikelihood(VecchiaApproximation, smodel, data;
                                    conditioning=KNNConditioning(15))


# Now we can do things like this: compute coupled simulations of a normal Matern
# kernel (sim_weaktailkernel) and a "fat" Matern kernel with α=1 above. Try
# plotting both sample paths together against getindex.(pts, 1)!
z   = randn(length(pts))
sim_weaktailkernel = simulate(spectral_nll, [1.0, 0.1, 1.4, 2.0]; z)
sim_fattailkernel  = simulate(spectral_nll, [1.0, 0.1, 1.4, 1.0]; z)

