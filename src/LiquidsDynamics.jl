module LiquidsDynamics


### Imports

# Load submodules code
include("Projections.jl")
include("LiquidsStructureGrids.jl")

# Load the actual modules
using Reexport
using Parameters
using LinearAlgebra: I

using .Projections
@reexport using .LiquidsStructureGrids


### Exports
export asymptotics, dynamics, dynamics!, initialize_dynamics


### Implementation
include("utils.jl")
include("initialization.jl")
include("globaldynamics.jl")
include("shorttimes.jl")
include("intermediatetimes.jl")
include("asymptotics.jl")


function __init__()
    # Hack to work around some performance issues in base (#32552, #28683)
    # Run a method and eval a part of the function that uses dynamic dispatch

    structure = StructureFactor(DipolarHardSpheres(0.3, 0.5), MSA{VerletWeis})
    grid = ChebyshevGrid(0, 50, 2^9)
    S = StructureFactorGrid(structure, grid)
    dynamics(S, 7.2, n = 16)

    @eval begin
        Projections.:*(v::TR, p::MDProjections{2}) =
            (@inbounds TR(v.t * (p.t + p.r[1]), v.r * p.r[2]))
    end
end


end # module
