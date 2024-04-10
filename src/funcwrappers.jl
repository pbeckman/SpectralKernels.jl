
struct SingularSDF{F}
  fn::F 
  p::Float64
end
(ssdf::SingularSDF{F})(w) where{F} = (w^ssdf.p)*ssdf.fn(w)

struct ParametricFun{F,P}
  fn::F
  p::P
end
(pfun::ParametricFun{F,P})(x) where{F,P} = pfun.fn(x, pfun.p)

struct ComponentFunction{J,F,P,X}
  f::F
  p::P
  x::X
end

struct ComponentDerivative{J,F}
  f::F
end

@generated function splice(x::NTuple{N,T}, newval::G, idx::Val{J}) where{J,N,T,G}
  quote
    ($([:(x[$j]) for j in 1:(J-1)]...), newval, $([:(x[$j]) for j in (J+1):N]...))
  end
end

@generated function splice(x::SVector{N,T}, newval::G, idx::Val{J}) where{J,N,T,G}
  quote
    @SVector [$([:(x[$j]) for j in 1:(J-1)]...), newval, $([:(x[$j]) for j in (J+1):N]...)]
  end
end

function (k::ComponentFunction{J,F,P,X})(pj) where{J,F,P,X}
  k.f(k.x, splice(k.p, pj, Val(J)))
end

function (dk::ComponentDerivative{J,F})(x::X, p::P) where{J,F,X,P}
  cf = ComponentFunction{J,F,P,X}(dk.f, p, x)
  ForwardDiff.derivative(cf, p[J])
end

@generated function component_derivatives(f::F, ::Val{N}) where{F,N}
  quote
    Base.Cartesian.@ntuple $N j->ComponentDerivative{j,F}(f)
  end
end
