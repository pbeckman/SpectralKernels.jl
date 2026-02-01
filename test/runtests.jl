
using Test, SpectralKernels

@testset "All tests" begin
    include("exponential_sdf_1d.jl")
    include("matern_sdf.jl")
end

