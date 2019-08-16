struct StaticVars{T, U, V, W, X, Y}
    S ::Vector{T} # structure factor
    Sˢ::Ones{U}   # self structure factor
    B ::Vector{V}
    Bˢ::W
    w ::Vector{X} # k-space integration weights
    υ ::Y         # scalar integration weight
end

struct KernelStaticVars{T, U}
    svars::T
    Λ::Vector{U}
end

struct DynamicsVars{T, U, V}
    F ::T # intermediate scattering function
    Fˢ::U # self intermediate scattering function
    ζ ::V # memory function
end

struct DynamicsAuxVars{T, U, V, W, X}
    A ::Vector{T}
    Aˢ::Vector{U}
    Z ::Vector{V}
    Fᵢ ::Vector{T}
    Fˢᵢ::Vector{U}
    ΔF₁ ::Vector{T}
    ΔFˢ₁::Vector{U}
    Δζ  ::Vector{W}
    ζoζ ::X
end

struct DynamicalProperties{T}
    #D ::Vector{T} # diffusion coefficient
    #W ::Vector{T} # mean square displacement
    Dₗ::Ref{T}    # long-time diffusion coefficient limit
end

struct DynamicsOutput{T, U <: DynamicsVars}
    τ::T
    dvars::U
    n::Int # number of time-grid points per decade
end

function DynamicsOutput(vars, grid, k, Δτ, n)
    τ  = collect(Δτ * (1:n))
    F  = collect(interpolate(grid, vars.F[i, :], k) for i = 1:n)
    Fˢ = collect(interpolate(grid, vars.Fˢ[i, :], k) for i = 1:n)
    ζ  = copy(vars.ζ)
    #Dₗ = Ref(inv(1 + Δτ * sum(Δζ)))

    dvars = DynamicsVars(F, Fˢ, ζ)
    #oprops = OtherDynamicalProperties(#=D, W, =#Dₗ)

    return DynamicsOutput{typeof(τ), typeof(dvars)}(τ, dvars, n)
end

function update!(output, vars, grid, k, Δτ, n₀, n)
    @unpack τ, dvars = output
    @unpack F, Fˢ, ζ = dvars

    ζₙ = view(vars.ζ, n₀:n)

    append!(τ, Δτ * (n₀:n))
    append!(F, (interpolate(grid, vars.F[i, :], k) for i = n₀:n))
    append!(Fˢ, (interpolate(grid, vars.Fˢ[i, :], k) for i = n₀:n))
    append!(ζ, ζₙ)
    #oprops.Dₗ[] = inv(inv(oprops.Dₗ[]) + Δτ * sum(Δζₙ))

    return output
end

function initialize_dynamics(structure, n)
    @unpack f, grid = structure
    liquid = f.liquid

    # · Precompute all the time-independent properties

    D₀ = diffusion_coeff(liquid)
    K  = nodes(grid)
    S  = project.(f.(K))
    Sˢ = onesof(S)
    Bˢ = bsfactors(D₀, K, S)
    B  = bfactors(Bˢ, S)
    d  = dimensionality(liquid)
    w  = weights(K, S, d, grid)
    υ  = weight(D₀, liquid.η, d, grid)
    Λ  = lambdaof(liquid).(K)

    svars = StaticVars(S, Sˢ, B, Bˢ, w, υ)
    kvars = KernelStaticVars(svars, Λ)

    # · Initialize the quantities we aim to compute

    m  = length(S)
    Fˢ = zeros(eltype(Sˢ), n, m)
    F  = zeros(eltype(S), n, m)
    ζ  = fill(zero(υ), n)
    #D   = copy(ζ)

    dvars = DynamicsVars(F, Fˢ, ζ)

    # · Preallocate auxiliar arrays

    A  = F[1, :]
    Aˢ = Fˢ[1, :]
    Z  = similar(w)
    Fᵢ  = copy(A)
    Fˢᵢ = copy(Aˢ)
    ΔF₁  = copy(A)
    ΔFˢ₁ = copy(Aˢ)
    Δζ   = copy(ζ)
    ζoζ  = init_conv(ζ)

    auxvars = DynamicsAuxVars(A, Aˢ, Z, Fᵢ, Fˢᵢ, ΔF₁, ΔFˢ₁, Δζ, ζoζ)

    return dvars, kvars, auxvars
end
