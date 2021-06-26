module Projections


### Imports
using Reexport
using LinearAlgebra: LinearAlgebra, UniformScaling
@reexport using StaticArrays


### Exports
export DProjections, LDProjections, MDProjections, TR, TvR, TRv,
       anyisless, anyiszero, checksizes, eachisless, getr, gett,
       llist, lptype, product, project, reduce_dof, ⋅

const ⋅ = LinearAlgebra.dot


### Implemetation

# Structs
abstract type AbstractProjections{T} end
abstract type AbstractTR end

# This is a struct that stores each projection (in spherical harmonics space)
# of a correlation function (`F` and `Fs`, for instance) for the case when the
# projections correlation matrix is diagonal in both `l` and `m`, i.e.,
# `F_{l,m;l',m'} = F_{l,m} * δ_{l,l'} * δ_{m,m'}`
struct DProjections{N, T} <: AbstractProjections{T}
    t::T
    r::SVector{N, T}
end

# This is a struct that carries information that depends only on each the
# l-values.  An important example is the projections for `Fs`, which are
# independent of `m`.  Other example is the set `l * (l + 1) * D₀` for all `l`
struct LDProjections{N, T} <: AbstractProjections{T}
    t::T
    r::SVector{N, T}
end

struct MDProjections{N, T} <: AbstractProjections{T}
    t::T
    r::SVector{N, T}
end

# Object that carries and separates
# translational and rotational information
# of a physical quantity
struct TR{T} <: AbstractTR
    t::T
    r::T
end

# This is two-field struct were the
# first component stores `K.^2 .* D₀.t`,
# and the second stores `[j * (j + 1) * D₀.r for j in 0:l]`
struct TvR{T, N} <: AbstractTR
    t::Vector{T}
    r::SVector{N, T}
end

struct TRv{T, N} <: AbstractTR
    t::T
    r::SVector{N, T}
end

# Constructors
DProjections(t::Number ) = DProjections(t, SVector{0, typeof(t)}())
LDProjections(t::Number) = LDProjections(t, SVector{0, typeof(t)}())
MDProjections(t::Number) = MDProjections(t, SVector{0, typeof(t)}())

LDProjections(v::TRv) = LDProjections(v.t, v.r)

constructorname(::Type{P}) where {P <: DProjections } = DProjections
constructorname(::Type{P}) where {P <: LDProjections} = LDProjections
constructorname(::Type{P}) where {P <: MDProjections} = MDProjections
constructorname(p::AbstractProjections) = constructorname(typeof(p))

project(x) = x
project(t::Tuple{Number}) = DProjections(first(t))
project(t::Tuple) = DProjections(first(t), SVector(Base.tail(t)))

Base.getindex(v::TvR, i) = TRv(v.t[i], v.r)
Base.length(v::TvR) = length(v.t)
Base.size(v::TvR) = size(v.t)
@inline Base.iterate(v::TvR, i = 1) =
    (i % UInt) - 1 < length(v) ? (@inbounds v[i], i + 1) : nothing

Base.eltype(::Type{TvR{T, N}}) where {T, N} = TRv{T, N}
Base.eltype(::Type{P}) where {T, P <: AbstractProjections{T}} = T

gett(v) = v.t
gett(v::Number) = v
getr(v) = v.r
getr(v::Union{Number,SVector}) = v

function checksizes(::Type{<:LDProjections{L}},
                    ::Type{<:DProjections{N}}) where {L, N}
    @assert div((L + 1) * (L + 2), 2) == N + 1
end
function checksizes(::Type{<:MDProjections{M}},
                    ::Type{<:DProjections{N}}) where {M, N}
    l = div(M + 2, 2)
    @assert div(l * (l + 1), 2) == N + 1
end
function checksizes(::Type{<:MDProjections{M}},
                    ::Type{<:LDProjections{L}}) where {L, M}
    @assert 2L == M
end
checksizes(S::Type{<:DProjections}, T::Type{<:LDProjections}) = checksizes(T, S)
checksizes(S::Type{<:DProjections}, T::Type{<:MDProjections}) = checksizes(T, S)
checksizes(S::Type{<:LDProjections}, T::Type{<:MDProjections}) = checksizes(T, S)

Base.:+(u::TR, v::TR) = TR(u.t + v.t, u.r + v.r)

Base.:-(u::TR, v::TR) = TR(u.t - v.t, u.r - v.r)

Base.:*(x::Number, p::AbstractProjections) = constructorname(p)(x * p.t, x * p.r)
Base.:*(p::AbstractProjections, x::Number) = x * p
#
Base.:*(x::Number, v::TR) = TR(x * v.t, x * v.r)
Base.:*(v::TR, x::Number) = x * v
Base.:*(u::TR, v::TR) = TR(u.t * v.t, u.r * v.r)
#
Base.:*(v::TR, p::MDProjections{2}) = @inbounds TR(v.t * (p.t + p.r[1]), v.r * p.r[2])
Base.:*(p::AbstractProjections, v::TR) = v * p

Base.:/(p::AbstractProjections, x::Number) = constructorname(p)(p.t / x, p.r / x)
#
Base.:/(v::TR, x::Number) = TR(v.t / x, v.r / x)
#
⋅(u::TR, v::TR) = u.t * v.t + u.r * v.t

for op in (:+, :-, :*)
    for P in (:DProjections, :LDProjections, :MDProjections)
        @eval begin
            Base.$op(p::($P){0}, q::($P){0}) = $P( $op(p.t, q.t) )
            Base.$op(p::($P){N}, q::($P){N}) where {N} = $P( $op(p.t, q.t), broadcast($op, p.r, q.r) )
        end
    end
end

Base.:*(v::TRv, p::DProjections) = TR(v.t * p, DProjections(zero(v.t), v.r .* p.r))

Base.:*(p::LDProjections{0}, q::DProjections{0}) = DProjections(p.t * q.t)
function Base.:*(p::LDProjections{1}, q::DProjections{2})
    t = p.t * q.t
    @inbounds begin
        r₁ = p.r[1] * q.r[1]
        r₂ = p.r[1] * q.r[2]
    end
    return DProjections(t, SVector(r₁, r₂))
end
function Base.:*(p::LDProjections{L}, q::DProjections{N}) where {L, N}
    checksizes(typeof(p), typeof(q))
    t = p.t * q.t
    T = typeof(t)
    r = zero(MVector{N, T})
    R = 1:L
    @inbounds for i in R
        r[i] = p.r.v[i] * q.r.v[i]
    end
    j = 0
    for i = (L + 1):N
        j += 1
        r[i] = p.r.v[R[j]] * q.r.v[i]
        if j == length(R)
            j = 0
            R = R[2:end]
        end
    end
    return DProjections(t, SVector(r))
end
Base.:*(p::DProjections, q::LDProjections) = q * p

product(x, y, z) = x * y * z
product(p::MDProjections{0}, q::DProjections{0}, r::LDProjections{0}) =
    MDProjections(p.t * q.t * r.t)
#
function product(p::MDProjections{2}, q::DProjections{2}, r::LDProjections{1})
    t = p.t * q.t * r.t
    @inbounds begin
        r₁ = p.r[1] * q.r[1] * r.r[1]
        r₂ = p.r[2] * q.r[2] * r.r[1]
    end
    return MDProjections(t, SVector(r₁, r₂))
end
#
function product(p::MDProjections{M}, q::DProjections{N}, r::LDProjections{L}) where {M, N, L}
    checksizes(typeof(p), typeof(q))
    checksizes(typeof(q), typeof(r))
    t = p.t * q.t * r.t
    T = typeof(t)
    s = zero(MVector{M, T})
    @inbounds for l = 1:L
        i = div(l * (l + 1), 2)
        m = l + L
        s[l] = p.r[l] * q.r[i] * r.r[l]
        s[m] = p.r[m] * q.r[i + 1] * r.r[l]
    end
    return MDProjections(t, SVector(s))
end

for f in (:exp, :inv)
    @eval begin
        Base.$f(p::AbstractProjections) = constructorname(p)($f(p.t), broadcast($f, p.r))
    end
end

Base.:+(J::UniformScaling, v::TR) = TR(J.λ + v.t, J.λ + v.r)
Base.:+(v::TR, J::UniformScaling) = J + v

Base.:+(v::TR, p::DProjections{0})  = DProjections(v.t + p.t)
Base.:+(v::TR, p::DProjections)     = DProjections(v.t + p.t, v.r .+ p.r)
Base.:+(v::TR, p::LDProjections{0}) = LDProjections(v.t + p.t)
Base.:+(v::TR, p::LDProjections)    = LDProjections(v.t + p.t, v.r .+ p.r)

Base.:-(J::UniformScaling, v::TR) = TR(J.λ - v.t, J.λ - v.r)

Base.:^(v::TR, n::Integer) = TR(v.t^n, v.r^n)
Base.inv(v::TR) = TR(inv(v.t), inv(v.r))

Base.zero(::Type{TR{T}}) where {T} = (o = zero(T); TR(o, o))
Base.zero(v::TR{T}) where {T} = (o = zero(T); TR(o, o))

Base.zero(::Type{ DProjections{0, T}}) where {T} = DProjections(zero(T))
Base.zero(::Type{LDProjections{0, T}}) where {T} = LDProjections(zero(T))
Base.zero(::Type{ DProjections{N, T}}) where {N, T} = DProjections(zero(T), zero(SVector{N, T}))
Base.zero(::Type{LDProjections{N, T}}) where {N, T} = LDProjections(zero(T), zero(SVector{N, T}))
Base.zero(::Type{MDProjections{N, T}}) where {N, T} = MDProjections(zero(T), zero(SVector{N, T}))
Base.zero(p::AbstractProjections) = constructorname(p)(zero(p.t), zero(p.r))

Base.one(v::TR) = TR(one(v.t), one(v.r))
#
Base.one(::Type{LDProjections{0, T}}) where {T} = LDProjections(one(T))
Base.one(::Type{LDProjections{L, T}}) where {L, T} = LDProjections(one(T), ones(SVector{L, T}))
Base.one(p::AbstractProjections) = constructorname(p)(one(p.t), ones(p.r))

anyiszero(v) = iszero(v)
anyiszero(v::TR{<:Number}) = iszero(v.t) || iszero(v.r)

anyisless(u, v) = u < v
anyisless(u, v::TR{<:Number}) = u < v.t || u < v.r
anyisless(u::TR{<:Number}, v) = u.t < v || u.r < v
anyisless(u::TR{<:Number}, y::TR{<:Number}) = u.t < v.t || u.t < v.r

eachisless(u, v) = u < v
eachisless(u, v::TR{<:Number}) = u < v.t && u < v.r
eachisless(u::TR{<:Number}, v) = u.t < v && u.r < v
eachisless(u::TR{<:Number}, v::TR{<:Number}) = u.t < v.t && u.r < v.r

function Base.isapprox(u::TR{T}, v::TR{T}; atol::Real = 0, rtol::Real = Base.rtoldefault(T),
                       nans::Bool = false) where {T <: Number}
    return isapprox(u.t, v.t; atol = atol, rtol = rtol, nans = nans) &&
           isapprox(u.r, v.r; atol = atol, rtol = rtol, nans = nans)
end

Base.muladd(x::Number, p::DProjections{0}, q::DProjections{0}) = DProjections(muladd(x, p.t, q.t))
function Base.muladd(x::Number, p::DProjections{2}, q::DProjections{2})
    t = muladd(x, p.t, q.t)
    @inbounds begin
        r₁ = muladd(x, p.r[1], q.r[1])
        r₂ = muladd(x, p.r[2], q.r[2])
    end
    return DProjections(t, SVector(r₁, r₂))
end
Base.muladd(x::Number, p::LDProjections{0}, q::LDProjections{0}) = LDProjections(muladd(x, p.t, q.t))
function Base.muladd(x::Number, p::LDProjections{1}, q::LDProjections{1})
    t = muladd(x, p.t, q.t)
    @inbounds begin
        r₁ = muladd(x, p.r[1], q.r[1])
    end
    return LDProjections(t, SVector(r₁))
end

reduce_dof(v) = v
reduce_dof(v::TR) = v.t + v.r
reduce_dof(v::TRv) = LDProjections(v.t, v.t .+ v.r)

lptype(::Type{DProjections{0, T}}) where {T} = LDProjections{0, T}
lptype(::Type{DProjections{2, T}}) where {T} = LDProjections{1, T}
@generated function lptype(::Type{DProjections{N, T}}) where {N, T}
    L = div(isqrt(9 + 8N) - 3, 2)
    P = LDProjections{L, T}
    return :($P)
end

@generated function mptype(::Type{DProjections{N, T}}) where {N, T}
    l = div(isqrt(9 + 8N) - 3, 2)
    V = SVector{2l, T}
    return :($V)
end

llist(::Type{DProjections{0, T}}) where {T} = SVector{0, T}()
llist(::Type{DProjections{2, T}}) where {T} = SVector(T(2))

#lprod(l) = l * (l + 1)
#@generated function lplist(::Type{DProjections{N, T}}) where {N, T}
#    L = div(isqrt(9 + 8N) - 3, 2)
#    P = LDProjections{L, T}
#    quote
#        SVector(map(lprod, 1:$L)...)
#    end
#end
#llist(::T) where {T <: DProjections} = llist(T)
#llist(x) = zero(x)


end # module
