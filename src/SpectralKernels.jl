
module SpectralKernels

  using LinearAlgebra, Printf # stdlibs
  using FastGaussQuadrature, FINUFFT, StaticArrays, QuadGK, FastHankelTransform # not stdlibs
  using TimerOutputs # not stdlibs
  import SpecialFunctions: gamma, sinint, expint, besselj

  # for the autodiff
  using DifferentiationInterface

  export AdaptiveKernelConfig, kernel_values, ParametricFunction, SpectralModel

  const TIMER = TimerOutput()

  include("wrappers.jl")
  include("utils.jl")
  include("quadrature.jl")
  include("adaptive.jl")
  include("derivatives.jl")
  include("model.jl")

end 

