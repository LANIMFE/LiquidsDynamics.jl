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


end # module
