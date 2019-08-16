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

diffusion_coeff(::HardDisks) = 1
diffusion_coeff(::HardSpheres) = 1
diffusion_coeff(::DipolarHardSpheres) = TR(1, 1)

bsfactors(D₀, K, S) = D₀ .* K.^2
bsfactors(D₀::TR, K, S::Vector{T}) where {T} = TvR(D₀.t .* K.^2, D₀.r * llist(T))

bfactors(Bˢ, S) = Bˢ .* inv.(S)

function weights(K, S, d, grid)
    ws = ApproximationGrids.weights(grid)
    return ws .* K.^(d + 1) .* (1 .- inv.(S)).^2
end

function weights(K, S::Vector{U}, d, grid) where {T, U <: DProjections{0, T}}
    w = fill(MDProjections(zero(T)), length(K))
    ws = ApproximationGrids.weights(grid)
    
    for i in eachindex(K)
        t = ws[i] * K[i]^4 * (1 - inv(S[i].t))^2
        w[i] = MDProjections(t)
    end    
    
    return w
end

function weights(K, S::Vector{U}, d, grid) where {T, U <: DProjections{2, T}}
    w = fill(MDProjections(zero(T), zero(SVector{2, T})), length(K))
    ws = ApproximationGrids.weights(grid)
    
    for i in eachindex(K)
        Sᵢ = S[i]
        k² = K[i]^2
        wᵣ = ws[i] * k²
        wₜ = wᵣ * k²
        
        # Projections components of the weights
        t  = wₜ * (1 - inv(Sᵢ.t))^2
        r₀ = 3wₜ * (1 - inv(Sᵢ.r[1]))^2
        r₁ = 6wᵣ * ((Sᵢ.r[1] - 1) * inv(Sᵢ.r[2]))^2
        
        w[i] = MDProjections(t, SVector(r₀, r₁))
    end
    
    return w
end

#function weights(K, S::Vector{U}, d, grid) where {N, T, U <: DProjections{N, T}}
#    V = Projections.mptype(U) # SVector{M, T}
#    z = zero(V)
#    l = length(z)
#    v = MVector(z)
#    w = fill(MDProjections(zero(T), z), length(K))
#    ws = weights(grid)
#
#    for i in eachindex(K)
#        Sᵢ = S[i]
#        k² = K[i]^2
#        wᵣ = ws[i] * k²
#
#        # Projections components of the weights
#        t = wᵣ * k² * (1 - inv(Sᵢ.t))^2
#        for j = 1:l
#            α = wᵣ * (2j + 1)
#            v[j] = α * k² * (1 - inv(Sᵢ.r[j]))^2
#            v[j] = α * j * (j + 1) * ((Sᵢ.r[j] - 1) * inv(Sᵢ.r[j + l]))^2
#        end
#
#        w[i] = MDProjections(t, SVector(v))
#    end
#
#    return w
#end

function weight(D₀, η, d, grid)
    j = jacobian(grid) / (4d * π^(d - 2) * η)
    return D₀ * j / d
end

function weight(D₀::TR, η, d, approx)
    j = jacobian(approx) / (12π * η)
    return TR(D₀.t * j / 3, D₀.r * j / 4)
end

init_conv(ζ) = nothing
init_conv(ζ::Vector{TR{T}}) where {T} = zeros(T, length(ζ))
