const AsymptoticVars = DynamicsVars

function asymptotics(S; tol = sqrt(eps()))
    # Dynamical variables, memory kernel variables and auxiliar variables
    avars, kvars, Z = initialize_asymptotics(S)

    asymptotics!(avars, kvars, Z, S, tol)
end

function asymptotics!(avars, kvars, Z, S, tol)
    @unpack f, fˢ = avars
    @unpack svars, Λ = kvars
    @unpack S, Sˢ, B, w, υ = svars

    ζ∞ = υ * sum(w)
    ζ′ = 2ζ∞

    @inbounds while nonconvergent(ζ′, ζ∞, tol)
        γ = D₀ * inv(ζ∞)
        ζ′ = ζ∞

        for j in eachindex(Λ)
            Sₑ = memory_term(B[j], γ)
            f[j]  = ergodic_param(S[j], Sₑ)
            fˢ[j] = ergodic_param(Sˢ[j], Sₑ)
            Z[j] = product(w[j], f[j], fˢ[j])
        end

        ζ∞ = υ * sum(Z)
    end

    return AsymptoticVars(f, fˢ, ζ∞)
end

function initialize_asymptotics(structure)
    @unpack f, grid = structure
    liquid = f.liquid

    D₀ = diffusion_coeff(liquid)
    K  = nodes(grid)
    S  = project.(f.(K))
    Sˢ = onesof(S)
    B  = K.^2 ./ gett.(Λ)
    Bˢ = nothing
    d  = dimensionality(liquid)
    w  = weights(K, S, d, grid)
    υ  = weight(one(D₀), liquid.η, d, grid)
    Λ  = lambdaof(liquid).(K)

    svars = StaticVars(S, Sˢ, B, Bˢ, w, υ)
    kvars = KernelStaticVars(svars, Λ)

    m  = length(S)
    fˢ = zeros(eltype(Sˢ), m)
    f  = zeros(eltype(S), m)
    ζ∞ = nothing

    avars = AsymptoticVars(f, fˢ, ζ∞)

    Z = similar(w)

    return avars, kvars, Z
end

memory_term(β, γ) = β * γ
#
function memory_term(β, γ::TR)
    t = β * gett(γ)
    return TR(t, t + 2 * getr(γ))
end

ergodic_param(Sⱼ, Sₑ) = Sⱼ / (Sⱼ + Sₑ)
#
function ergodic_param(Sⱼ::DProjections{3}, Sₑ)
    t = Sⱼ.t / (Sⱼ.t + Sₑ.t)
    @inbounds begin
        r₁ = Sⱼ.r[1] / (Sⱼ.r[1] + Sₑ.r)
        r₂ = Sⱼ.r[2] / (Sⱼ.r[2] + Sₑ.r)
    end
    return DProjections(t, SVector(r₁, r₂))
end
#
function ergodic_param(Sⱼ::LDProjections{2}, Sₑ)
    t = Sⱼ.t / (Sⱼ.t + Sₑ.t)
    @inbounds begin
        r₁ = Sⱼ.r[1] / (Sⱼ.r[1] + Sₑ.r)
    end
    return LDProjections(t, SVector(r₁))
end
