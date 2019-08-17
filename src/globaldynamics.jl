"""    dynamics(S, k, t; n = 128, Δτ = 1e-7, tol = 1e-6)

Computes the dynamical properties of a liquid `S` with structure factor
approximated over a finite grid.  In particular it computes the intermediate
scattering function `F` and the self intermediate scattering function `Fs` at a
point `k` of the wavevector space as functions of the correlation time, up to a
time `t`.  It also computes the memory function `ζ` of the Generalized
Langevin Equation, which can later be used to estimate the diffusion
coefficient and the mean square displacement as functions of the correlation
time.

The algorithm is based on the theoretical model known as Self-consistent
Generalized Langevin Equation.  The precision of the method can be controlled
by the keyword arguments `n`, `Δτ` and `tol`, which correspond to the number of
points in the time grid per 'time decade', the initial time grid spacing, and
the convergence of the relative tolerance for the memory function `ζ`,
respectively.
"""
function dynamics(S, k, t; Δτ = 1e-7, n = 128, tol = sqrt(eps()))
    # Number of time points for which the short-times approximation is used
    @assert n ≥ (n₀ = 8)
    # Dynamical variables, memory kernel variables and auxiliar variables
    dvars, kvars, auxvars = initialize_dynamics(S, n)

    # The current function is just an initialization routine,
    # the real work starts here
    return dynamics!(dvars, kvars, auxvars, S, k, t, Δτ, n₀, n, tol)
end

function dynamics!(dvars, kvars, auxvars, S, k, t, Δτ, n₀, n, tol)
    # Use a short-times approximation initially
    approximate!(dvars, kvars.svars, auxvars.Z, Δτ, n₀)
    # Fill the first time grid employing the SCGGLE equations
    solve!(dvars, kvars, auxvars, Δτ, n₀ + 1, n, tol)

    output = DynamicsOutput(dvars, S.grid, k, Δτ, n)

    # As long as the target time `t` is not reached, keep doubling the time
    # grid spacing `Δτ` and solve for each new time point.  Since we decimate
    # the previous grid keeping the first-half information, we only need to
    # compute the dynamics for the second half of each new time grid.
    n₀ = div(n, 2) + 1
    #Dₗ = output.oprops.Dₗ
    while n * Δτ < t
        Δτ *= 2
        decimate!(dvars)
        solve!(dvars, kvars, auxvars, Δτ, n₀, n, tol)
        update!(output, dvars, S.grid, k, Δτ, n₀, n)
    end

    #write(output)
    return output
end
