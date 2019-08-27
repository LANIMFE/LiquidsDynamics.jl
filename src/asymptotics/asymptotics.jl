"""    asymptotics(S; rtol = sqrt(eps()))

Computes the ergodic parameters and asymptotic value of the memory kernel of
the Generalized Langevin Equation.  The precision of the method can be
controlled by the keyword argument `rtol`, which sets the relative tolerance
for asymptotic value ot the memory function, `ζ∞`.
"""
function asymptotics(S; rtol = sqrt(eps()))
    # Dynamical variables, memory kernel variables and auxiliar variables
    avars, kvars, D₀ = initialize_asymptotics(S)
    Z = similar(kvars.svars.w)

    return asymptotics!(avars, kvars, Z, S, D₀, avars.ζ∞.x, rtol)
end

function asymptotics!(avars, kvars, Z, S, D₀, ζ′, rtol)
    @unpack f, fˢ = avars
    @unpack svars, Λ = kvars
    @unpack S, Sˢ, B, w, υ = svars

    ζ∞ = ζ′
    ζ′ = 2ζ′

    @inbounds while isnonconvergent(ζ′, ζ∞, rtol)
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

    avars.ζ∞.x = ζ∞
    return avars
end

function asymptotic_mobility!(b, b′, ζ∞, dvars, kvars, auxvars, Δτ, n₀, n, rtol, atol)
    if !isanyzero(ζ∞.x)
        return b.x = zero(b.x)
    end

    while isnonconvergent(b′.x, b.x, rtol)
        Δτ *= 2
        b′ = b
        decimate!(dvars)
        solve!(dvars, kvars, auxvars, Δτ, n₀, n, rtol)
        b.x = b.x + integrate(Δτ, view(dvars.ζ, n₀:n))
        if isanyless(inv(b.x), atol)
            break
        end
    end

    return b.x = inv(b.x)
end
