module LiquidsDynamics


### Imports
using Reexport
using Parameters
using Projections
@reexport using LiquidsStructureGrids
using LinearAlgebra: I


### Exports
export dynamics, dynamics!, initialize_dynamics


### Implementation
include("utils.jl")
include("initialization.jl")
include("globaldynamics.jl")
include("shorttimes.jl")
include("intermediatetimes.jl")


function __init__()
    # Hack to work around some performance issues in base (#32552, #28683)
    # Run a method and eval a part of the function that uses dynamic dispatch

    structure = StructureFactor(DipolarHardSpheres(0.3, 0.05), MSA{VerletWeis})
    grid = ChebyshevGrid(0, 50, 2^9)
    S = StructureFactorGrid(structure, grid)
    dynamics(S, 7.2, 1e6, n = 16)

    @eval Projections begin
        *(v::TR, p::MDProjections{2}) =
            (@inbounds TR(v.t * (p.t + p.r[1]), v.r * p.r[2]))
    end
end


end # module
