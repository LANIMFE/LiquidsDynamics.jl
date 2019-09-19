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

# Similar to `Ref` from base, but with some useful extras
mutable struct Box{T}
    x::T

    Box{T}() where {T} = new()
    Box{T}(x) where {T} = new(x)
end

Box(x::T) where {T} = Box{T}(x)

Base.getindex(b::Box) = b.x
Base.setindex!(b::Box, x) = (b.x = x; b)

function Base.show(io::IO, ::MIME"text/plain", b::Box)
    if get(io, :compact, false)
        print(io, "Box")
    end
    print(io, "(", b.x, ")")
end

Base.one(b::Box) = Box(one(b.x))


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

diff!(ΔV, V, i) = (ΔV[i] = V[i - 1] - V[i])

conv!(VoV::Nothing, V, n) = nothing
function conv!(VoV::Vector{T}, V, n) where {T}
    nₕ = n ÷ 2
    VoVn = zero(T)
    for i = 1:nₕ
        VoVn += (V[i].t * V[n - i + 1].r) + (V[i].r * V[n - i + 1].t)
    end
    if isodd(n)
        VoVn += V[nₕ + 1].t * V[nₕ + 1].r
    end
    return VoV[n] = VoVn
end

# Simpson integration. The lower endpoint can be optionally supplied through
# the keyword argument `y₀`.
function integrate(h, y::AbstractVector{T}; y₀::Union{T, Nothing} = nothing) where {T}
    haskeyword = y₀ !== nothing
    i₁ = Int(!haskeyword)
    n = length(y)
    @inbounds begin
        y₁ = haskeyword ? y₀ : y[1]
        I  = sum(y[i] for i = (i₁ + 3):(n - 3))
        I += (23 // 24) * (y[i₁ + 2] + y[n - 2])
        I += ( 7 // 6 ) * (y[i₁ + 1] + y[n - 1])
        I += ( 3 // 8 ) * (y₁ + y[n])
    end
    return h * I
end
