module LiquidsStructureGrids


### Imports
using Reexport
@reexport using ApproximationGrids
@reexport using LiquidsStructure

import ApproximationGrids: ApproximationGrid


### Implemetation

# Types
"""
    Ones{T}

Indexable object that returns `one(T)` when indexed.
"""
struct Ones{T} <: AbstractArray{T, 0} end

struct StructureFactorGrid{S <: StructureFactor, G <: ApproximationGrid}
    f::S
    grid::G
end

# Constructors
Ones(::AbstractArray{T}) where {T} = Ones{T}()

# Methods
Base.size(::Ones) = () # Important for broadcasting purposes
Base.getindex(::Ones{T}, I...) where {T} = one(T)
Base.checkbounds(::Ones, I...) = nothing

Base.show(io::IO, ::MIME"text/plain", A::Ones) = print(io, typeof(A), "()")
function Base.show(io::IO, ::MIME"text/plain", s::StructureFactorGrid)
    print(io, "$(s.f),$(s.grid)")
end


### Exports
export Ones, StructureFactorGrid


end # module
