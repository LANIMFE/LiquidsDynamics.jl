struct AsymptoticVars{T, U, V}
    f ::T
    fˢ::U
    ζ∞::V
end

function initialize_asymptotics(structure)
    @unpack f, grid = structure
    liquid = f.liquid

    D₀ = diffusion_coeff(liquid)
    K  = nodes(grid)
    S  = project.(f.(K))
    Sˢ = onesof(S)
    d  = dimensionality(liquid)
    w  = weights(Asymptotics, K, S, d, grid)
    υ  = weight(one(D₀), liquid.η, d, grid)
    Λ  = lambdaof(liquid).(K)
    B  = K.^2 ./ gett.(Λ)
    Bˢ = nothing

    svars = StaticVars(S, Sˢ, B, Bˢ, w, υ)
    kvars = KernelStaticVars(svars, Λ)

    m  = length(S)
    fˢ = zeros(eltype(Sˢ), m)
    f  = zeros(eltype(S), m)
    ζ∞ = Box(υ * sum(w))

    avars = AsymptoticVars(f, fˢ, ζ∞)

    return avars, kvars, D₀
end
