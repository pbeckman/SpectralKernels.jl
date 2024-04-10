module SpectralKernels

  using LinearAlgebra, Printf # stdlibs
  using FastGaussQuadrature, FINUFFT, StaticArrays, Vecchia, ForwardDiff, QuadGK # not stdlibs
  using FastChebInterp, TimerOutputs, MLStyle # not stdlibs
  import SpecialFunctions: gamma, sinint, expint

  export AdaptiveKernelConfig, kernel_values

  const TIMER = TimerOutput()

  include("utils.jl")
  include("quadrature.jl")
  include("adaptive.jl")
  include("funcwrappers.jl")
  include("kernel.jl")
  include("vecchia.jl")

end 
