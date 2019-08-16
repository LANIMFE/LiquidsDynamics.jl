module LiquidsStructureGrids


### Imports
using Reexport
@reexport using ApproximationGrids
@reexport using LiquidsStructure

import ApproximationGrids: ApproximationGrid


### Exports
export StructureFactorGrid


### Implemetation
struct StructureFactorGrid{S <: StructureFactor, G <: ApproximationGrid}
    f::S
    grid::G
end

function Base.show(io::IO, ::MIME"text/plain", s::StructureFactorGrid)
    print(io, "$(s.f),$(s.grid)")
end


end # module
