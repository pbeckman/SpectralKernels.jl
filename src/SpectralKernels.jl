module SpectralKernels

  using LinearAlgebra, Printf # stdlibs
  using FastGaussQuadrature, FINUFFT, StaticArrays, ForwardDiff, QuadGK, FastHankelTransform # not stdlibs
  using TimerOutputs # not stdlibs
  import SpecialFunctions: gamma, sinint, expint, besselj0

  export AdaptiveKernelConfig, kernel_values

  const TIMER = TimerOutput()

  include("utils.jl")
  include("quadrature.jl")
  include("adaptive.jl")
  include("funcwrappers.jl")

end 
