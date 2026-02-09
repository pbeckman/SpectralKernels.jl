
using SpectralKernels

include("matern_pair.jl")

# a standard stationary Matern model, but using the new parameterization. No
# scale parameter for now, that is another special case. Note that this spectral
# density takes its parameters as SCALAR arguments, and the frequency argument
# MUST come first.
iso_sdf(w, nu) = (one(w) + w^2)^(-nu - 1/2)

# The warping function, on the other hand, takes warping parameters as a single
# VECTOR. This is because the parameter count for warp_params may be high, and
# because we can get away with doing so without losing AD backend flexibility.
warp(warp_params, x) = x/warp_params[1] + abs((x - cos(2*x)))^2

# Specify how you want to integrate the spectral density.
cfg = AdaptiveKernelConfig(iso_sdf; tol=1e-12)

# Let's actually pick some points and work through a full example. Let's say
# that we're on a regular lattice on [0,1]. 
xgrid = collect(range(0.0, 5.0, length=250))

# At building time, we have to make a decision about how a flat vector of
# parameters will get parsed. We'll use the following partition:
#
# params = vcat(sdf_params, warp_params) = [nu, rho].
#
# So then sdf_param_indices=(1,) (turning 1:1 into a tuple) and
# warp_param_indices=(2,).
#
# The actual covariance function that is evaluated is:
#
# K_iso(h, params) 
#   = \int_{R^d} cos(2*pi*h^T w) S(w, params...) dw,
#
# K(x, y, (sdf_params, warp_params)) 
#   = K_iso( warp(warp_params, x) - warp(warp_params, y), iso_params).
#
model = SpectralModel(;cfg=cfg, warp=warp, sdf_param_indices=(1,),
                      warp_param_indices=(2,), singularity_param_index=0,
                      pts=xgrid, pts_pairs=vec(collect(Iterators.product(1:250, 1:250))))

# now for specific parameters, we can generate a kernel object.
kern = SpectralKernels.gen_kernel(model, (2.5, 1.0))

# Now we can treat this like a normal function and make a matrix as usual.
M = [kern(xj, xk) for xj in xgrid, xk in xgrid]

# TODO (cg 2026/02/09 13:33): why is this necessary??? lag 0 has some issues.
M[1,1] /= 2

sim = cholesky(Symmetric(M)).L*randn(250, 3)

