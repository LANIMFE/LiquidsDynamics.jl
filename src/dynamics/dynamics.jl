"""    dynamics(S, k; Δτ = 1e-7, t = 1e7, n = 128, rtol = sqrt(eps()), brtol = eps())

Computes the dynamical properties of a liquid `S` with structure factor
approximated over a finite grid.  In particular it computes the intermediate
scattering function `F` and the self intermediate scattering function `Fs` at
wavenumber `k` as functions of the correlation time, up to a time `t` (`1e7` by
default).  It also computes the memory function `ζ` of the Generalized Langevin
Equation, which can later be used to estimate the diffusion coefficient and the
mean square displacement as functions of the correlation time.

The algorithm is based on the theoretical model known as Self-consistent
Generalized Langevin Equation.  The precision of the method can be controlled
by the keyword arguments `n`, `Δτ` and `rtol`, which correspond to the number
of points in the time grid per 'time decade', the initial time grid spacing,
and the convergence of the relative tolerance for the memory function `ζ`,
respectively.

This function also returns the long-time-limit mobility `b` of the system up to
a relative tolerance specified by `brtol`.
"""
function dynamics(S, k; t = 1e7, Δτ = 1e-7, n = 128, rtol = sqrt(eps()), brtol = eps())
    # Number of time points for which the short-times approximation is used.
    @assert n ≥ (n₀ = 8)
    # Dynamical variables, memory kernel variables and auxiliar variables.
    dvars, kvars, auxvars = initialize_dynamics(S, n)

    # The current function is just an initialization routine,
    # the real work starts here.
    return dynamics!(dvars, kvars, auxvars, S, k, t, Δτ, n₀, n, rtol, brtol)
end

function dynamics!(dvars, kvars, auxvars, S, k, t, Δτ, n₀, n, rtol, brtol)
    # Use a short-times approximation initially.
    approximate!(dvars, kvars.svars, auxvars.Z, Δτ, n₀)
    # Fill the first time grid employing the SCGGLE equations.
    solve!(dvars, kvars, auxvars, Δτ, n₀ + 1, n, rtol)
    # Allocate the output with the solution for the first-decade time grid.
    #
    # TODO: This could be improved moving the following line to `dynamics`,
    #       but the size should be carefully allocated.
    output = DynamicsOutput(dvars, S.grid, k, Δτ, n)

    # As long as the target time `t` is not reached, keep doubling the time
    # grid spacing `Δτ` and solve for each new time point.  Since we decimate
    # the previous grid keeping the first-half information, we only need to
    # compute the dynamics for the second half of each new time grid.
    n₀ = div(n, 2) + 1
    b′ = one(output.b)
    while 2n * Δτ < t
        Δτ *= 2
        b′ = output.b
        decimate!(dvars)
        solve!(dvars, kvars, auxvars, Δτ, n₀, n, rtol)
        update!(output, dvars, S.grid, k, Δτ, n₀, n)
    end

    # Reaching the time `t` won't generally guarantee that the asymptotic value
    # for the mobility `b` has converged.  If the asymptotic value of the
    # memory kernel is finite and greater than zero, we know that `b` should
    # the be zero, so we use this fact.  Otherwise, we keep iterating until
    # convergence of `b` up to relative precision `brtol`, but no longer store
    # the solutions to the SCGLE.
    avars, akvars, D₀ = initialize_asymptotics(S)
    g = ζ -> fixedpoint!(auxvars.Z, akvars, D₀, ζ)
    asymptotics!(g, avars, last(dvars.ζ); rtol = rtol)
    asymptotic_mobility!(output.b, b′, avars.ζ∞, dvars, kvars, auxvars,
                         Δτ, n₀, n, rtol, brtol)

    return output
end
