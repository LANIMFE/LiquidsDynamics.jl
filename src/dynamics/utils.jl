### Decimation for dynamic-variables objects
function decimate!(vars::DynamicsVars)
    decimate!(vars.F)
    decimate!(vars.Fˢ)
    decimate!(vars.ζ)
end


### Static auxiliar quantities
function dynamics_weights!(w, ws, K, S, d, grid)
    w .= ws .* K.^(d + 1) .* (1 .- inv.(S)).^2
    return w
end
#
function dynamics_weights!(w::Vector{U}, ws, K, S, d, grid) where {T, U <: DProjections{0, T}}

    @inbounds for i in eachindex(K)
        t = ws[i] * K[i]^4 * (1 - inv(S[i].t))^2
        w[i] = MDProjections(t)
    end

    return w
end
#
function dynamics_weights!(w::Vector{U}, ws, K, S, d, grid) where {U <: MDProjections{2}}

    @inbounds for i in eachindex(K)
        Sᵢ = S[i]
        k² = K[i]^2
        wᵣ = ws[i] * k²
        wₜ = wᵣ * k²

        # Projections components of the weights
        t  = wₜ * (1 - inv(Sᵢ.t))^2
        r₀ = 3wₜ * (1 - inv(Sᵢ.r[1]))^2
        r₁ = 6wᵣ * ((Sᵢ.r[1] - 1) * inv(Sᵢ.r[2]))^2

        w[i] = MDProjections(t, SVector(r₀, r₁))
    end

    return w
end
#
#function dynamics_weights!(w::Vector{U}, ws, K, S, d, grid) where {U <: MDProjections}
#
#    @inbounds for i in eachindex(K)
#        Sᵢ = S[i]
#        k² = K[i]^2
#        wᵣ = ws[i] * k²
#
#        # Projections components of the weights
#        t = wᵣ * k² * (1 - inv(Sᵢ.t))^2
#        for j = 1:l
#            α = wᵣ * (2j + 1)
#            v[j] = α * k² * (1 - inv(Sᵢ.r[j]))^2
#            v[j] = α * j * (j + 1) * ((Sᵢ.r[j] - 1) * inv(Sᵢ.r[j + l]))^2
#        end
#
#        w[i] = MDProjections(t, SVector(v))
#    end
#
#    return w
#end
