
using Test, SpectralKernels
using SpectralKernels.DifferentiationInterface
using SpectralKernels.LinearAlgebra
using SpectralKernels.Printf

@testset "Core functionality" begin
    include("exponential_sdf_1d.jl")
    include("matern_sdf.jl")
end

using ForwardDiff

@testset "Derivatives" begin
  include("derivatives/warping.jl")
end

