
using SpectralKernels, DifferentiationInterface, ForwardDiff

# specify the model and the kernel, and then the warping function.
iso_sdf(w, power)  = exp(-abs(w)^power) # power ∈ [1,2], say.
warp(params, x)    = abs((x/params[1])^params[2])

sdf_params  = (1.3,) # must be a tuple, because the iso_sdf must accept scalar args!
warp_params = [inv(50.0), 1.1]
backend     = AutoForwardDiff()


fn(w, a, b, c) = a^2 + b*w^c
sdf   = SpectralKernels.ParametricSDF(fn, (1.0, 2.0, 3.0))

#=
cfg   = SpectralKernels.AdaptiveKernelConfig(sdf; tol=1e-12)
xgrid = vcat(collect(range(0.001, 1.1, length=20)))
=#

