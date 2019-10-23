"""    asymptotics(S; rtol = eps())

Computes the ergodic parameters and asymptotic value of the memory kernel of
the Generalized Langevin Equation.  The precision of the method can be
controlled by the keyword argument `rtol`, which sets the relative tolerance
for asymptotic value ot the memory function, `ζ∞`.
"""
function asymptotics(S; rtol = eps(), verbose = false)
    # Dynamical variables, memory kernel variables and auxiliar variables
    avars, kvars, D₀ = initialize_asymptotics(S)
    Z = similar(kvars.svars.w)

    g = ζ -> fixedpoint!(Z, kvars, D₀, ζ)

    return asymptotics!(g, avars, avars.ζ∞[], rtol, verbose = verbose)
end

function asymptotics!(g, avars, ζ∞, rtol; verbose = false)
    # Initialize Anderson Acceleration (AA). We use only the previous point,
    # that is, we set `m = 1` in the AA algorithm.
    # TODO: Try AA with `m = 2`.
    z₋ = ζ∞
    z̃₋ = g(z₋)
    f₋ = z₋ - z̃₋
    z = z̃₋
    z̃ = g(z)
    f = z - z̃

    # Counter to keep track of how many times the sequence regresses whenever
    # `f₋, f` are positive.
    c = 0
    # AA step and reference value to check when the sequence regresses.
    Δz = Δz̃ = zero(z)
    # A trust region is used to ensure positive-definitennes of the solution.
    trustregion = EllipsoidalRegion()

    while true
        if iszero(f) || isapprox(z, z̃; rtol = rtol, nans = true)
            break
        end

        Δf = f - f₋
        if iszero(Δf)
            Δz = zero(Δz)
        else
            γ = (Δf ⋅ f) / (Δf ⋅ Δf)
            Δz = bound(trustregion, z̃, γ * (z̃₋ - z̃))
        end

        if eachisless(0, Δz) && eachisless(0, f) && eachisless(0, f₋)
            if c < 1
                Δz̃ = Δz
            elseif (c ≥ 1 && eachisless(Δz̃, Δz)) || c > 8
                z = f = zero(z)
                break
            end
            c += 1
        end
    end

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
