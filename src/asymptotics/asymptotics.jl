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

    return asymptotics!(avars, kvars, Z, D₀, avars.ζ∞[], rtol)
end

function asymptotics!(avars, kvars, Z, D₀, ζ∞, rtol)
    @unpack f, fˢ = avars
    @unpack svars, Λ = kvars
    @unpack S, Sˢ, B, w, υ = svars

    @inbounds while true
        γ = D₀ * inv(ζ∞)
        ζ′ = ζ∞

        for j in eachindex(Λ)
            Sₑ = memory_term(B[j], γ)
            f[j]  = ergodic_param(S[j], Sₑ)
            fˢ[j] = ergodic_param(Sˢ[j], Sₑ)
            Z[j] = product(w[j], f[j], fˢ[j])
        end

        ζ∞ = υ * sum(Z)

        if isapprox(ζ′, ζ∞; rtol = rtol, nans = true)
            break
        end
    end

    avars.ζ∞[] = ζ∞
    return avars
end

function asymptotic_mobility!(b, b′, ζ∞, dvars, kvars, auxvars, Δτ, n₀, n, rtol, brtol)
    if !isanyzero(ζ∞[])
        return b[] = zero(b[])
    end

    while true
        Δτ *= 2
        b′ = b

        decimate!(dvars)
        solve!(dvars, kvars, auxvars, Δτ, n₀, n, rtol)
        b[] = b[] + integrate(Δτ, view(dvars.ζ, (n₀ - 1):n))

        if isapprox(b′[], b[]; rtol = brtol, nans = true)
            break
        end
    end

    return b[] = inv(b[])
end
