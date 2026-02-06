
using Test, SpectralKernels
using SpectralKernels.DifferentiationInterface

@testset "Core functionality" begin
    include("exponential_sdf_1d.jl")
    include("matern_sdf.jl")
end

using ForwardDiff

@testset "Derivatives" begin
  include("derivatives/warping.jl")
end

