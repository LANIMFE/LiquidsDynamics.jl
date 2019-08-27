### Lazy objects that act as vectors of ones
"""    Ones{T}

Indexable object that returns `one(T)` when indexed.
"""
struct Ones{T} <: AbstractArray{T, 0} end

# Indirect constructors
onesof(::AbstractArray{T}) where {T} = Ones{T}()
onesof(::AbstractVector{P}) where {P <: DProjections} = Ones{lptype(P)}()

# Methods
Base.eltype(::Ones{T}) where {T} = T
Base.size(::Ones) = () # Important for broadcasting purposes
Base.getindex(::Ones{T}, I...) where {T} = one(T)
Base.checkbounds(::Ones, I...) = nothing

Base.show(io::IO, ::MIME"text/plain", A::Ones) = print(io, typeof(A), "()")


mutable struct Mutable{T}
    x::T
end

function Base.show(io::IO, ::MIME"text/plain", m::Mutable)
    if get(io, :compact, false)
        print(io, "Mutable")
    end
    print(io, "(", m.x, ")")
end

Base.one(m::Mutable) = Mutable(m.x)


### Interpolation function for the SCGLE theory
struct Lambda{T}
    kᶜ::T

    Lambda{T}(k::T) where {T} = new(k)
end

Lambda(α) = (kᶜ = 2π * α; Lambda{typeof(kᶜ)}(kᶜ))

# Indirect constructors from LiquidsStructure.Liquid subtypes
lambdaof(::HardSpheres) = Lambda(1.305)
lambdaof(::DipolarHardSpheres) = Lambda(TR(1.305, Inf))

(λ::Lambda)(k) = inv(I + (k * inv(λ.kᶜ))^2)


### Diffusion coefficents
diffusion_coeff(::HardDisks) = 1
diffusion_coeff(::HardSpheres) = 1
diffusion_coeff(::DipolarHardSpheres) = TR(1, 1)


### Static auxiliar quantities
bsfactors(D₀, K, S) = D₀ .* K.^2
bsfactors(D₀::TR, K, S::Vector{T}) where {T} = TvR(D₀.t .* K.^2, D₀.r * llist(T))

bfactors(Bˢ, S) = Bˢ .* inv.(S)

function weights(E, K, S, d, grid)
    w = alloc_weights(K, S)
    ws = ApproximationGrids.weights(grid)
    return weights!(E, w, ws, K, S, d, grid)
end

alloc_weights(K, S) = fill(zero(eltype(S)), length(K))
alloc_weights(K, S::Vector{U}) where {T, U <: DProjections{2, T}} =
    fill(zero(MDProjections{2, T}), length(K))

weights!(::Type{Dynamics}, w, ws, K, S, d, grid) =
    dynamics_weights!(w, ws, K, S, d, grid)
weights!(::Type{Asymptotics}, w, ws, K, S, d, grid) =
    asymptotics_weights!(w, ws, K, S, d, grid)

function weight(D₀, η, d, grid)
    j = jacobian(grid) / (4d * π^(d - 2) * η)
    return D₀ * j / d
end
#
function weight(D₀::TR, η, d, approx)
    j = jacobian(approx) / (12π * η)
    return TR(D₀.t * j / 3, D₀.r * j / 4)
end

init_conv(ζ) = nothing
init_conv(ζ::Vector{TR{T}}) where {T} = zeros(T, length(ζ))


### Other utils
function decimate!(V::Vector)
    m = length(V)
    @inbounds for i = 2:2:m
        V[div(i, 2)] = (V[i - 1] + V[i]) / 2
    end
    return V
end

function decimate!(M::Matrix)
    m = size(M, 1)
    for i = 2:2:m
        @views M[div(i, 2), :] .= (M[i, :] .+ M[i - 1, :]) ./ 2
    end
    return M
end

isnonconvergent(ζᵢ, ζ, tol) = abs(1 - ζᵢ / ζ) > tol
isnonconvergent(ζᵢ::TR, ζ, tol) = (abs(1 - ζᵢ.t / ζ.t) > tol || abs(1 - ζᵢ.r / ζ.r) > tol)

diff!(Δζ, ζ, i) = (Δζ[i] = ζ[i - 1] - ζ[i])

conv!(ζoζ::Nothing, ζ, n) = nothing
function conv!(ζoζ::Vector{T}, ζ, n) where {T}
    nₕ = n ÷ 2
    ζoζn = zero(T)
    for i = 1:nₕ
        ζoζn += (ζ[i].t * ζ[n - i + 1].r) + (ζ[i].r * ζ[n - i + 1].t)
    end
    if isodd(n)
        ζoζn += ζ[nₕ + 1].t * ζ[nₕ + 1].r
    end
    return ζoζ[n] = ζoζn
end

# TODO: Try with Simpson's rule instead of trapezoid rule
function integrate(h, v)
    n = length(v)
    @inbounds begin
        Σ = sum(v[i] for i = 2:n) + (v[1] + v[end]) / 2
    end
    return h * Σ
end
