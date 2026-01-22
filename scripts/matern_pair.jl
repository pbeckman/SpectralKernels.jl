
using BesselK, HypergeometricFunctions

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
function sing_matern_cov(t, p, parms)
  (v, a, b) = parms
  t*a > 2 && @warn @sprintf("sing_matern_cov is known to be unstable when t*ρ > 2. You evaluated it with t = %.2e, ρ = %.2e", t, a) maxlog=1
  return v * a^(-2b)*(2pi)^(-p) / BesselK._gamma(0.5 + b) * (
      2^(2b+1)*pi^(2b)*(a*t)^(2b)*t^(-p)*cos(b*pi-p*pi/2)*BesselK._gamma(0.5+b)*BesselK._gamma(-2b+p) * 
      pFq((0.5+b,), (0.5+b-p/2, 1+b-p/2), a^2*pi^2*t^2) + 
      (2pi)^p*(a)^p*BesselK._gamma(b-p/2)*BesselK._gamma(0.5+p/2) * 
      pFq((0.5+p/2,), (0.5, 1-b+p/2), a^2*pi^2*t^2)
  )
end
sing_matern_cov(t, params) = sing_matern_cov(t-0.0, params[end], params[1:(end-1)])