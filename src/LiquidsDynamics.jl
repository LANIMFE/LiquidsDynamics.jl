module LiquidsDynamics


### Imports

# Load submodules code
include("Projections.jl")
include("LiquidsStructureGrids.jl")

# Load the actual modules
using Reexport
using Parameters
using LinearAlgebra: I

@reexport using .Projections
@reexport using .LiquidsStructureGrids


### Exports
export asymptotics, asymptotics!, dynamics, dynamics!, initialize_asymptotics,
       initialize_dynamics


### Implementation
abstract type SCGLE end
abstract type Dynamics    <: SCGLE end
abstract type Asymptotics <: SCGLE end

include("utils.jl")
include("dynamics/dynamics.jl")
include("dynamics/init.jl")
include("dynamics/shorttimes.jl")
include("dynamics/intermediatetimes.jl")
include("dynamics/utils.jl")
include("asymptotics/asymptotics.jl")
include("asymptotics/init.jl")
include("asymptotics/utils.jl")


function __init__()
    # Hack to work around some performance issues in base (#32552, #28683)
    # Run a method and eval a part of the function that uses dynamic dispatch

    structure = StructureFactor(DipolarHardSpheres(0.3, 1.0), MSA{VerletWeis})
    grid = ChebyshevGrid(0, 50, 2^9)
    S = StructureFactorGrid(structure, grid)
    dynamics(S, 7.2, n = 16)

    @eval begin
        Projections.:*(v::TR, p::MDProjections{2}) =
            (@inbounds TR(v.t * (p.t + p.r[1]), v.r * p.r[2]))
    end
end


end # module
