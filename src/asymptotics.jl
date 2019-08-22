struct AsymptoticVars{T, U, V}
    f ::T
    fˢ::U
    ζ∞::V
end

function asymptotics(S; tol = sqrt(eps()))
    # Dynamical variables, memory kernel variables and auxiliar variables
    avars, kvars, D₀ = initialize_asymptotics(S)

    υ  = kvars.svars.υ
    w  = kvars.svars.w
    Z  = similar(w)
    ζ′ = υ * sum(w)

    asymptotics!(avars, kvars, Z, S, D₀, ζ′, tol)
end

function asymptotics!(avars, kvars, Z, S, D₀, ζ′, tol)
    @unpack f, fˢ = avars
    @unpack svars, Λ = kvars
    @unpack S, Sˢ, B, w, υ = svars

    ζ∞ = ζ′
    ζ′ = 2ζ′

    @inbounds while is_nonconvergent(ζ′, ζ∞, tol)
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

function asymptotic_D!(D, D′, ζ∞, dvars, kvars, auxvars, Δτ, n₀, n, rtol, atol)
    if !any_iszero(ζ∞)
        return D.b = zero(D.b)
    end

    while is_nonconvergent(D′, D.b, rtol)
        Δτ *= 2
        D′ = D.b
        decimate!(dvars)
        solve!(dvars, kvars, auxvars, Δτ, n₀, n, rtol)
        D.b = D.b + trapz(Δτ, view(dvars.ζ, n₀:n))
        if any_isless(inv(D.b), atol)
            break
        end
    end

    return D.b = inv(D.b)
end

function initialize_asymptotics(structure)
    @unpack f, grid = structure
    liquid = f.liquid

    D₀ = diffusion_coeff(liquid)
    K  = nodes(grid)
    S  = project.(f.(K))
    Sˢ = onesof(S)
    d  = dimensionality(liquid)
    w  = asymptotics_weights(K, S, d, grid)
    υ  = weight(one(D₀), liquid.η, d, grid)
    Λ  = lambdaof(liquid).(K)
    B  = K.^2 ./ gett.(Λ)
    Bˢ = nothing

    svars = StaticVars(S, Sˢ, B, Bˢ, w, υ)
    kvars = KernelStaticVars(svars, Λ)

    m  = length(S)
    fˢ = zeros(eltype(Sˢ), m)
    f  = zeros(eltype(S), m)
    ζ∞ = nothing

    avars = AsymptoticVars(f, fˢ, ζ∞)

    return avars, kvars, D₀
end

function asymptotics_weights(K, S, d, grid)
    ws = ApproximationGrids.weights(grid)
    return ws .* K.^(d + 1) .* (S .- 1).^2 .* inv.(S)
end
#
function asymptotics_weights(K, S::Vector{U}, d, grid) where
    {T, U <: DProjections{2, T}}

    w = fill(MDProjections(zero(T), zero(SVector{2, T})), length(K))
    ws = ApproximationGrids.weights(grid)

    @inbounds for i in eachindex(K)
        Sᵢ = S[i]
        k² = K[i]^2
        wᵣ = ws[i] * k²
        wₜ = wᵣ * k²

        # Projections components of the weights
        t  = wₜ * (Sᵢ.t - 1)^2 * inv(Sᵢ.t)
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
