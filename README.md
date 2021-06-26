# LiquidsDynamics.jl

[![GitHub Actions](https://github.com/LANIMFE/LiquidsDynamics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/LANIMFE/LiquidsDynamics.jl/actions?query=workflow%3ACI)

This library intended to provide a mean to compute the intermediate scattering function
<img src="https://render.githubusercontent.com/render/math?math=F(k,t)"> and its self
component from the Self Consistent Generalized Langevin Equation (SCGLE) formalism in the
[Julia](http://julialang.org) programming language.

## Installation

`LiquidsDynamics.jl` is compatible with Julia 1.0 and later versions. It requires first
adding the [LANIMFE-Registy](https://github.com/LANIMFE/LANIMFE-Registry) to your Julia
installation. Then it can simply be installed by running

```julia
julia> ]
pkg> add LiquidsDynamics
```

## Acknowledgements

This project was developed with support from CONACYT through the Laboratorio
Nacional de Ingeniería de la Materia Fuera de Equilibrio (LANIMFE).
