
using Test, SpectralKernels, ForwardDiff
using SpectralKernels.DifferentiationInterface
using SpectralKernels.LinearAlgebra
using SpectralKernels.Printf

include("../scripts/matern_pair.jl")

@testset "Derivatives" begin
  include("derivatives/warping.jl")
  include("derivatives/argswap.jl")
  include("derivatives/sdf_params.jl")
  include("derivatives/jacobian.jl")
end

#=
@testset "Core functionality" begin
    include("exponential_sdf_1d.jl")
    include("matern_sdf.jl")
end
=#

