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
