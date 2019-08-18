const ErgodicParams = DynamicsVars

function ergodic_parameters(S; tol = sqrt(eps()))
    # Dynamical variables, memory kernel variables and auxiliar variables
    output, kvars, Z = initialize_asymptotics(S)

    ergodic_parameters!(output, kvars, Z, S, tol)
end

function ergodic_parameters!(output, kvars, Z, S, tol)
    @unpack f, fˢ = output

    ζ∞ = υ * sum(w)
    ζ′ = 2ζ∞

    @inbounds while nonconvergent(ζ′, ζ∞, tol)
        γ = D₀ * inv(ζ∞)
        ζ′ = ζ∞

        for j in eachindex(Λ)
            fₑ = ergodic_common(B, γ, j)
            f[j]  = ergodic_param(S[j], fₑ)
            fˢ[j] = ergodic_param(Sˢ[j], fₑ)
            Z[j] = product(w[j], f[j], fˢ[j])
        end

        ζ∞ = υ * sum(Z)
    end

    return ErgodicParams(f, fˢ, ζ∞)
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

    output = ErgodicParams(f, fˢ, ζ∞)

    Z = similar(w)

    return output, kvars, Z
end
