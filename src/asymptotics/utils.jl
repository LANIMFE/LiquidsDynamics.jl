abstract type TrustRegion end

struct RectangularRegion <: TrustRegion end
struct EllipsoidalRegion <: TrustRegion end

const Φ = 1 / MathConstants.φ

function bound(::TrustRegion, z, Δz)
    if Δz > z
        return Φ * z
    end
    return Δz
end
#
function bound(::RectangularRegion, z::TR, Δz)
    steeper = Δz.r * z.t - Δz.t * z.r > 0
    if Δz.r > z.r && steeper
        zʳ = Φ * z.r
        return TR((Δz.t / Δz.r) * zʳ, zʳ)
    end
    if Δz.t > z.t && !steeper
        zᵗ = Φ * z.t
        return TR(zᵗ, (Δz.r / Δz.t) * zᵗ)
    end
    return Δz
end
#
function bound(::EllipsoidalRegion, z::TR, Δz)
    if iszero(z.r)
        return TR(min(Δz.t, Φ * z.t), z.r)
    end
    if iszero(z.t)
        return TR(z.t, min(Δz.r, Φ * z.r))
    end
    zᵗ = Δz.t / z.t
    zʳ = Δz.r / z.r
    r = hypot(zᵗ, zʳ)
    if r > 1
        return TR(z.t * (zᵗ / r), z.r * (zʳ / r))
    end
    return Δz
end

function asymptotics_weights!(w, ws, K, S, d, grid)
    # We write `(1 .- inv.(S)).^2 .* S` instead of `(S .- 1).^2 .* inv.(S)`
    # to get `w[j] == Inf` instead of `NaN` whenever `S[j] == Inf`.
    w .= ws .* K.^(d + 1) .* (1 .- inv.(S)).^2 .* S
    return w
end
#
function asymptotics_weights!(w::Vector{U}, ws, K, S, d, grid) where
    {T, U <: MDProjections{2, T}}

    @inbounds for i in eachindex(K)
        Sᵢ = S[i]
        k² = K[i]^2
        wᵣ = ws[i] * k²
        wₜ = wᵣ * k²

        # Projections components of the weights
        t  = wₜ * (1 - inv(Sᵢ.t))^2 * Sᵢ.t
        r₀ = (Sᵢ.r[1] - 1)^2
        r₁ = 3wₜ * r₀ * inv(Sᵢ.r[1])
        r₂ = 6wᵣ * r₀ * inv(Sᵢ.r[2])

        w[i] = MDProjections(t, SVector(r₁, r₂))
    end

    return w
end

memory_term(β, γ) = β * γ
#
function memory_term(β, γ::TR)
    t = β * gett(γ)
    return TR(t, t + 2 * getr(γ))
end

ergodic_param(Sⱼ, Sₑ) = Sⱼ / (Sⱼ + Sₑ)
#
function ergodic_param(Sⱼ::DProjections{2}, Sₑ)
    t = Sⱼ.t / (Sⱼ.t + Sₑ.t)
    @inbounds begin
        r₁ = Sⱼ.r[1] / (Sⱼ.r[1] + Sₑ.r)
        r₂ = Sⱼ.r[2] / (Sⱼ.r[2] + Sₑ.r)
    end
    return DProjections(t, SVector(r₁, r₂))
end
#
function ergodic_param(Sⱼ::LDProjections{1}, Sₑ)
    t = Sⱼ.t / (Sⱼ.t + Sₑ.t)
    @inbounds begin
        r₁ = Sⱼ.r[1] / (Sⱼ.r[1] + Sₑ.r)
    end
    return LDProjections(t, SVector(r₁))
end

function set!(avars::AsymptoticVars, Z, kvars, D₀, ζ)
    @unpack f, fˢ = avars
    @unpack svars, Λ = kvars
    @unpack S, Sˢ, B, w, υ = svars

    γ = D₀ * inv(ζ)

    @inbounds for j in eachindex(Λ)
        Sₑ = memory_term(B[j], γ)
        f[j] = ergodic_param(S[j], Sₑ)
        fˢ[j] = ergodic_param(Sˢ[j], Sₑ)
    end

    avars.ζ∞[] = ζ
    return avars
end

struct FixedPoint{U, V, W} <: Function
    Z::U
    kvars::V
    D₀::W
end
#
(f::FixedPoint)(ζ) = fixedpoint!(f.Z, f.kvars, f.D₀, ζ)

function fixedpoint!(Z, kvars, D₀, ζ)
    @unpack svars, Λ = kvars
    @unpack S, Sˢ, B, w, υ = svars

    γ = D₀ * inv(ζ)

    @inbounds for j in eachindex(Λ)
        Sₑ = memory_term(B[j], γ)
        f = ergodic_param(S[j], Sₑ)
        fˢ = ergodic_param(Sˢ[j], Sₑ)
        Z[j] = product(w[j], f, fˢ)
    end

    return υ * sum(Z)
end
