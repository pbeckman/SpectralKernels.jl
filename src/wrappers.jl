
struct ParametricFunction{F,P}
  fn::F
  params::NTuple{P,Float64}
end

(psdf::ParametricFunction{F,P})(w::Float64) where{F,P} = psdf.fn(w, psdf.params...)

struct ArgSwap{F,P,J}
  fn::ParametricFunction{F,P} 
end

function (as::ArgSwap{F,P,J})(args...) where{F,P,J}
  swapped_args = ntuple(j->(j==1 ? args[J] : (j==J ? args[1] : args[j])), P+1)
  as.fn.fn(swapped_args...)
end

# Unlike ParametricFunction, this is an internal object.
struct ParametricDerivative{F,P,J,A,B}
  swap_sdf::ArgSwap{F,P,J}
  prep::A
  backend::B
end

function ParametricDerivative(psdf::ParametricFunction{F,P}, ::Val{J}, 
                              backend::B) where{F,P,J,B}
  swap = ArgSwap{F,P,J+1}(psdf) # J+1 since args[1] is the frequency.
  prep = prepare_derivative(swap, backend, 1.0, Constant.(psdf.params)...)
  ParametricDerivative(swap, prep, backend)
end

# So here is the annoying thing: you can only differentiate with respect to the
# first arg in DifferentiationInterface.jl, so it falls on us to write an
# annoying wrapper to secretly permute the args.
#
# What I _want_ is to say
#
# prep = prepare_derivative(sdf::ParametricFunction, backend, Constant(w),
#                           [Constants], params[j], [Constants])
#
# But the only allowable syntax is
#
# prep = prepare_derivative(sdf::ParametricFunction, backend, params[j], [Constants]).
#
# So this whole difficult-to-read apparatus above and in this routine here is to
# make a struct wrapper that swaps the first and J-th args, makes a plan for
# that, and then differentiates the swapped functions **with swapped
# parameters**, which is a double swap and ends up differentiating with respect
# to the right J-th parameter.
function (pd::ParametricDerivative{F,P,J,A,B})(w) where{F,P,J,A,B}
  args          = (w, pd.swap_sdf.fn.params...)
  permuted_args = ntuple(j->(j==1 ? args[J] : Constant((j==J ? w : args[j]))), P+1)
  derivative(pd.swap_sdf, pd.prep, pd.backend, permuted_args...)
end

