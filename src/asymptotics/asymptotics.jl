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

    return asymptotics!(g, avars, avars.ζ∞[]; rtol = rtol, verbose = verbose)
end

function asymptotics!(g, avars, ζ; rtol = rtol, verbose = false)
    # Initialize Anderson Acceleration (AA). We use only the previous point,
    # that is, we set `m = 1` in the AA algorithm.
    # TODO: Try AA with `m = 2`.
    z₋ = ζ
    z̃₋ = g(ζ)
    f₋ = z₋ - z̃₋
    z = z̃₋
    z̃ = g(z)
    f = z - z̃

    # Counter to keep track of how many times the sequence regresses whenever
    # `f₋, f` are positive.
    c = 0
    # AA step and reference value to check when the sequence regresses.
    Δz = Δz̃ = zero(z)
    # A trust region is used to ensure positive-definiteness of the solution.
    trustregion = EllipsoidalRegion()

    while true
        if iszero(f) || isapprox(z, z̃; rtol = rtol, nans = true)
            break
        end

        Δf = f - f₋
        Δf² = Δf ⋅ Δf
        if iszero(Δf²)
            Δz = zero(Δz)
        else
            γ = (Δf ⋅ f) / Δf²
            Δz = bound(trustregion, z̃, γ * (z̃ - z̃₋))
        end

        if eachisless(Δz, 0) && eachisless(0, f) && eachisless(0, f₋)
            if c < 1
                Δz̃ = Δz
            elseif (c ≥ 1 && eachisless(Δz, Δz̃)) || c > 16
                z = f = zero(z)
                break
            end
            c += 1
        end

        z₋ = z
        z̃₋ = z̃
        f₋ = f
        z = z̃ - Δz
        z̃ = g(z)
        f = z - z̃

        verbose && @show z, f
    end

    return set_ergodic_params!(avars, g.Z, g.kvars, g.D₀, z)
end

function asymptotic_mobility!(b, b′, ζ, dvars, kvars, auxvars, Δτ, n₀, n, rtol, brtol)
    if !anyiszero(ζ[])
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
