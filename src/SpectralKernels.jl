
module SpectralKernels

  using LinearAlgebra, Printf # stdlibs
  using FastGaussQuadrature, FINUFFT, StaticArrays, QuadGK, FastHankelTransform # not stdlibs
  using TimerOutputs # not stdlibs
  import SpecialFunctions: gamma, sinint, expint, besselj

  # for the autodiff
  using DifferentiationInterface

  export AdaptiveKernelConfig, kernel_values

  const TIMER = TimerOutput()

  include("utils.jl")
  include("quadrature.jl")
  include("adaptive.jl")
  include("derivatives.jl")

end 

