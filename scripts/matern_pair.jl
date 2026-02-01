
using BesselK, HypergeometricFunctions, Printf

# Isotropic Matern covariance with parameters p evaluated at distance t
function matern_cov(t, p; d=1)
  (phi, alpha, v) = p
    constant  = pi^(d/2)*phi
    constant /= (2^(v-1))*BesselK._gamma(v+d/2)*alpha^(2*v)
    arg = alpha*2pi*abs(t)
    iszero(arg) && return constant*BesselK.adbesselkxv(v, 0)
    constant*BesselK.adbesselk(v, arg)*(arg^v)
end

# Matern spectral density
matern_sdf(w, p; d=1) = p[1]*(p[2]^2 + w^2)^(-p[3] - d/2)

# Singular Matern kernel with singularity power p=-α
function sing_matern_cov(t, p, parms; d=1)
  (phi, a, b) = parms
  t*a > 2 && @warn @sprintf("sing_matern_cov is known to be unstable when t*ρ > 2. You evaluated it with t = %.2e, ρ = %.2e", t, a) maxlog=1
  out  = pi^p * (a*t)^p * gamma((d + p)/2) * 
    pFq(((d + p)/2,), (d/2, (2 - 2b + p)/2), a^2*pi^2*t^2) / 
    (gamma(d/2) * gamma((2 - 2b + p)/2))
  out -= pi^(2b) * (a*t)^(2b) * gamma(b + d/2) * 
    pFq((b + d/2,), (1 + b - p/2, b + d/2 - p/2), a^2*pi^2*t^2) /
    (gamma(1 + b - p/2) * gamma(b + d/2 - p/2))
  out *= phi * a^(-2b) * pi^(1+d/2-p) * t^(-p) * csc(b*pi-(p*pi)/2) / gamma(b+d/2)

  return out
end
sing_matern_cov(t, params) = sing_matern_cov(t-0.0, params[end], params[1:(end-1)])