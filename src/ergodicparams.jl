function ergodic_parameters(S; tol = sqrt(eps()))
    # Dynamical variables, memory kernel variables and auxiliar variables
    avars, kvars, = initialize_asymptotics(S)

    ergodic_parameters!(avars, kvars, S, tol)
end

function ergodic_parameters!(avars, kvars, S, tol)
    B  = K.^2 ./ ᵀ.(Λ)
    W  = ergodic_weights(K, S, d, approx) # LDProjections
    w  = ergodic_weight(D₀, η, d, approx) # TRVec

    f  = fill(zero(T), m)
    fˢ = fill(llist(T), m)

    ft  = fill(zero(eltype(W)), m) # MDProjections

    Δζ∞ = reduce_sum(w, W)
    Δζ′ = 2Δζ∞

    @inbounds while nonconvergent(Δζ′, Δζ∞, tol)
        γ = D₀ * inv(Δζ∞)
        Δζ′ = Δζ∞

        for j in eachindex(Λ)
            fₑ = ergodic_common(B, γ, j)
            f[j]  = ergodic_param(S[j], fₑ)
            fˢ[j] = ergodic_param(Sˢ[j], fₑ)
            ft[j] = deltazeta_step(W[j], f[j], fˢ[j])
        end

        Δζ∞ = reduce_sum(w, ft)
    end

    return f, fˢ, Δζ∞
end

function initialize_asymptotics(structure)
    @unpack f, grid = structure
    liquid = f.liquid

    D₀ = diffusion_coeff(liquid)
    K  = nodes(grid)
    S  = project.(f.(K))
    Sˢ = onesof(S)
    d  = dimensionality(liquid)
    Λ  = lambdaof(liquid).(K)

    m  = length(S)
end
