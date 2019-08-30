"""    approximate!(dvars, svars, Z, Δτ, n)

Assigns to the i-th columns of `F`, for `i = 1:n` the short-time limit of the
intermediate scattering function, and calculates the memory function for the
same time-points.
"""
function approximate!(dvars, svars, Z, Δτ, n)
    @unpack F, Fˢ, ζ = dvars
    @unpack S, Sˢ, B, Bˢ, w, υ = svars

    @inbounds for i = 1:n
        τ = i * Δτ

        for j in eachindex(S)
            F[i, j]  = short_times_limit!(S[j], B[j], τ)
            Fˢ[i, j] = short_times_limit!(Sˢ[j], Bˢ[j], τ)
            Z[j] = product(w[j], F[i, j], Fˢ[i, j])
        end

        ζ[i] = υ * sum(Z)
    end

    return dvars
end

short_times_limit!(Sⱼ, β, τ) = exp(-τ * reduce_dof(β)) * Sⱼ
