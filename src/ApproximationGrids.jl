module ApproximationGrids


### Imports
using FFTW


### Implemetation
abstract type ApproximationGrid end


# Structs
struct ChebyshevGrid{T <: AbstractFloat, P} <: ApproximationGrid
    a::T
    b::T
    n::Int
    p::P
end

struct RegularGrid{T <: AbstractFloat} <: ApproximationGrid
    a::T
    b::T
    n::Int
end

# Constructors
function ChebyshevGrid(a::A, b::B, n) where {A, B}
    T = promote_type(float(A), float(B))
    p = FFTW.plan_r2r!(fill(zero(T), n), FFTW.REDFT10)
    return ChebyshevGrid{T, typeof(p)}(a, b, n, p)
end

function RegularGrid(a::A, b::B, n) where {A, B}
    T = promote_type(float(A), float(B))
    return RegularGrid{T}(a, b, n)
end

# Nodes
function nodes(g::ChebyshevGrid{T}) where {T}
    n = g.n
    a = T(0.5) * (g.b - g.a)
    b = T(0.5) * (g.b + g.a)
    return [a * sinpi((2k + one(T) - n) / 2n) + b for k = 0:(n - 1)]
end

function nodes(g::RegularGrid{T}) where {T}
    n = g.n
    h = (g.b - g.a) / (n - 1)
    return g.a .+ h * (0:(n - 1))
end

# Weights
function weights(g::ChebyshevGrid)
    n = g.n
    a = zeros(n)
    a[1:2:n] = (2 / n) ./ (1 .- ((1:2:n) .- 1).^2)
    return FFTW.r2r(a, FFTW.REDFT01)
end

function weights(g::RegularGrid)
    @assert g.n ≥ 4
    m = g.n % 3

    w = fill(oftype(g.a, 9 // 8), g.n)
    w[1] = 3 // 8
    for i = 4:3:(g.n - 3)
        w[i] = 3 // 4
    end
    if m == 1
        w[end    ] =  3 // 8
    elseif m == 0
        w[end - 2] = 17 // 24
        w[end - 1] =  4 // 3
        w[end    ] =  1 // 3
    elseif m == 2
        w[end - 3] =  7 // 6
        w[end - 2] = 11 // 12
        w[end - 1] =  7 // 6
        w[end    ] =  3 // 8
    end

    return w
end

# Jacobian
jacobian(g::ChebyshevGrid{T}) where {T} = T(0.5) * (g.b - g.a)
jacobian(g::RegularGrid) = (g.b - g.a) / (g.n - 1)

# Interpolation
function interpolate(g::ChebyshevGrid, v::AbstractVector, x′)
    @assert (g.a ≤ x′ ≤ g.b)
    x = 2 * (x′ - (g.b + g.a) / 2) / (g.b - g.a)

    n = length(v)

    c = reverse_plan(g.p, v, n)
    c[1] /= 2
    c   ./= n

    if n == 0
        return zero(eltype(c))
    elseif n == 1 # avoid issues with NaN x
        return first(c) * one(x)
    end

    x = 2x
    b₁ = b₂ = zero(eltype(c))
    @inbounds for k = n:-1:2
        b₂, b₁ = b₁, muladd(x, b₁, c[k] - b₂)
    end

    return muladd(x / 2, b₁, c[1] - b₂)
end
            
reverse_plan(p, v::AbstractVector{Float64}, n) = p * reverse(v)
function reverse_plan(p, v::AbstractVector{S}, n) where {S}
    r = reinterpret(eltype(S), reverse(v))
    m = length(r)
    d = div(m, n)
    w = permutedims(reshape(r, d, n))
    
    for i in 1:d
        p * view(w, :, i) # `p` is an in-place `FFTW` plan which mutates `w`
        r[i:d:m] = view(w, :, i)
    end
    
    return reinterpret(S, r)
end

function interpolate(g::RegularGrid, v::AbstractVector, x′)
    @assert (g.a ≤ x′ ≤ g.b)
    r = x′ .- range(g.a, stop = g.b, length = g.n)

    i = findfirst(x -> x ≤ 0, r)
    if r[i] == 0
        return v[i] / one(r[i]) # in case there are type instabilities
    end
    k = 3 * div(i - 1, 3)
    k = (k + 4 > g.n) ? (g.n - 4) : k

    # Barycentric interpolation weights
    w₁ =  1 / r[k + 1]
    w₂ = -3 / r[k + 2]
    w₃ =  3 / r[k + 3]
    w₄ = -1 / r[k + 4]
    # Interpolation function reference values
    v₁, v₂, v₃, v₄ = v[k + 1], v[k + 2], v[k + 3], v[k + 4]

    return (w₁ * v₁ + w₂ * v₂ + w₃ * v₃ + w₄ * v₄) / (w₁ + w₂ + w₃ + w₄)
end

# Objects' string representations
function Base.show(io::IO, g::T) where {T <: ApproximationGrid}
    print(io, "$(nameof(T))(($(g.a),$(g.b)), $(g.n))")
end


### Exports
export ChebyshevGrid, RegularGrid,
       interpolate, jacobian, nodes, weights


end # module
