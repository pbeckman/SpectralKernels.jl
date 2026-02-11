
# Now in 2D!

using SpectralKernels, StaticArrays

include("matern_pair.jl")

# a standard stationary Matern model, but using the new parameterization. No
# scale parameter for now, that is another special case. Note that this spectral
# density takes its parameters as SCALAR arguments, and the frequency argument
# MUST come first.
iso_sdf(w, nu) = (one(w) + norm(w)^2)^(-nu - 1)

# The warping function, on the other hand, takes warping parameters as a single
# VECTOR. This is because the parameter count for warp_params may be high, and
# because we can get away with doing so without losing AD backend flexibility.
warp(warp_params, x) = SA[x[1]/warp_params[1], hypot(x[1], x[2])/warp_params[2]]

# Let's actually pick some points and work through a full example. Let's say
# that we're on a regular lattice on [0,1]. 
g1d = range(0.0, 1.0, length=8)
pts = vec(SVector{2,Float64}.(Iterators.product(g1d, g1d)))

# At building time, we have to make a decision about how a flat vector of
# parameters will get parsed. We'll use the following partition:
#
# params = vcat(sdf_params, warp_params) = [nu, rho1, rho2].
#
# So then sdf_param_indices=(1,) (turning 1:1 into a tuple) and
# warp_param_indices=(2,3).
#
# The actual covariance function that is evaluated is:
#
# K_iso(h, params) 
#   = \int_{R^d} cos(2*pi*h^T w) S(w, params...) dw,
#
# K(x, y, (sdf_params, warp_params)) 
#   = K_iso( warp(warp_params, x) - warp(warp_params, y), iso_params).
#
model = SpectralModel(iso_sdf, pts; warp=warp, sdf_param_indices=1,
                      warp_param_indices=[2,3], tol=1e-12)

# now for specific parameters, we can generate a kernel object.
kern = SpectralKernels.gen_kernel(model, (2.5, 1.0, 0.1))

# Now we can treat this like a normal function and make a matrix as usual.
M = [kern(xj, xk) for xj in pts, xk in pts]

# TODO (cg 2026/02/09 13:33): Weirdly, when both inputs are zero,
# K(warp(0)-warp(0)) has a specific correctness issue. Maybe something I
# introduced in refactoring.
M[1,1] = M[2,2]

sim = cholesky(Symmetric(M)).L*randn(length(pts), 3)

