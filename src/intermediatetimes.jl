function decimate!(V::Vector)
    m = length(V)
    @inbounds for i = 2:2:m
        V[div(i, 2)] = (V[i - 1] + V[i]) / 2
    end
    return V
end

function decimate!(M::Matrix)
    m = size(M, 1)
    for i = 2:2:m
        @views M[div(i, 2), :] .= (M[i, :] .+ M[i - 1, :]) ./ 2
    end
    return M
end

function decimate!(vars::DynamicsVars)
    decimate!(vars.F)
    decimate!(vars.Fˢ)
    decimate!(vars.ζ)
end

function solve!(dvars, kvars, auxvars, Δτ, n₀, n, tol)
    @unpack F, Fˢ, ζ = dvars
    @unpack svars, Λ = kvars
    @unpack S, Sˢ, B, Bˢ, w, υ = svars
    @unpack A, Aˢ, Z, Fᵢ, Fˢᵢ, ΔF₁, ΔFˢ₁, Δζ, ζoζ = auxvars

    #đ = inv(1 / Δτ + Δζ[1])
    fill_aux_svars!(A, Aˢ, ΔF₁, ΔFˢ₁, S, Sˢ, Bˢ, Λ, F, Fˢ, ζ[1], Δτ)
    fill_aux_tvars!(Δζ, ζoζ, ζ, n₀ - 1)

    for i = n₀:n
        perform_linstep!(Fᵢ, Fˢᵢ, F, Fˢ, ΔF₁, ΔFˢ₁, B, Bˢ, ζ, Δζ, ζoζ, Λ, Δτ, i)
        perform_nlstep!(F, Fˢ, Fᵢ, Fˢᵢ, ΔF₁, ΔFˢ₁, A, Aˢ, B, Bˢ, Z, ζ, Λ, w, υ, Δτ, i, tol)
        diff!(Δζ, ζ, i)
        conv!(ζoζ, ζ, i)
        #dynamical_properties!(D, Δζ, ζ, đ, Δτ, i)
    end

    return dvars
end

function fill_aux_svars!(A, Aˢ, ΔF₁, ΔFˢ₁, S, Sˢ, Bˢ, Λ, F, Fˢ, ζ₁, Δτ)
    Δτ⁻¹ = inv(Δτ)

    @inbounds for j in eachindex(S)
        ΔF₁[j]  = S[j] - F[1, j]
        ΔFˢ₁[j] = Sˢ[j] - Fˢ[1, j]

        m = Λ[j] * ζ₁
        a = alpha_a(Δτ, Δτ⁻¹, m)
        b = alpha_b(Bˢ[j], Δτ, m)
        A[j]  = inv(a + b * inv(S[j]))
        Aˢ[j] = inv(a + b)
    end

    return A, Aˢ, ΔF₁, ΔFˢ₁
end

alpha_a(Δτ, Δτ⁻¹, m) = (Δτ⁻¹ + m)
function alpha_a(Δτ, Δτ⁻¹, m::TR)
    t = Δτ⁻¹ + m.t
    r = t + m.r * (1 + Δτ * m.t)
    return TR(t, r)
end

alpha_b(k²D₀, Δτ, m) = k²D₀
function alpha_b(k²D₀::LDProjections{1}, Δτ, m::TR)
    t = k²D₀.t
    r = t * (1 + Δτ * m.r) + (1 + Δτ * m.t) * k²D₀.r[1]
    return LDProjections(t, SVector(r))
end
function alpha_b(k²D₀, Δτ, m::TR)
    t = k²D₀.t
    r = t * (1 + Δτ * m.r) .+ (1 + Δτ * m.t) .* k²D₀.r
    return LDProjections(t, r)
end

function fill_aux_tvars!(Δζ, ζoζ::Nothing, ζ, n)
    @inbounds for i = 2:n
        diff!(Δζ, ζ, i)
    end
    return Δζ
end

function fill_aux_tvars!(Δζ, ζoζ, ζ, n)
    conv!(ζoζ, ζ, 1)
    @inbounds for i = 2:n
        diff!(Δζ, ζ, i)
        conv!(ζoζ, ζ, i)
    end
    return Δζ
end

diff!(Δζ, ζ, i) = (Δζ[i] = ζ[i - 1] - ζ[i])

conv!(ζoζ::Nothing, ζ, n) = nothing
function conv!(ζoζ::Vector{T}, ζ, n) where {T}
    nₕ = n ÷ 2
    ζoζn = zero(T)
    for i = 1:nₕ
        ζoζn += (ζ[i].t * ζ[n - i + 1].r) + (ζ[i].r * ζ[n - i + 1].t)
    end
    if isodd(n)
        ζoζn += ζ[nₕ + 1].t * ζ[nₕ + 1].r
    end
    return ζoζ[n] = ζoζn
end

## Linear contributions of the SCGLE equations
function perform_linstep!(Fᵢ, Fˢᵢ, F, Fˢ, ΔF₁, ΔFˢ₁, B, Bˢ, ζ, Δζ, ζoζ, Λ, Δτ, n)
    n₋ = n - 1
    ζn₋ = ζ[n₋]

    @inbounds for j in eachindex(Λ)
        λ, β, βˢ, ΔF₁ⱼ, ΔFˢ₁ⱼ, Δτλλ = index_lin_svars(Λ, B, Bˢ, ΔF₁, ΔFˢ₁, Δτ, j)

        Fₐ  = initialize_linterm(F[1, j], ζn₋, ζoζ, n₋)
        Fˢₐ = initialize_linterm(Fˢ[1, j], ζn₋, ζoζ, n₋)

        for i = 2:n₋
            l = n - i + 1
            Δζₗ, Δτζₗ, ζᵢζₗ, Δζoζₗ = index_lin_tvars(Δζ, ζ, ζoζ, Δτ, l, i)

            Fₐ  = accumulate_linterm(Fₐ, F[i, j], Δζₗ, Δτζₗ, ζᵢζₗ, Δζoζₗ, β, ΔF₁ⱼ)
            Fˢₐ = accumulate_linterm(Fˢₐ, Fˢ[i, j], Δζₗ, Δτζₗ, ζᵢζₗ, Δζoζₗ, βˢ, ΔFˢ₁ⱼ)
        end

        Fᵢ[j]  = reduce_linterm(Fₐ, λ, F[n₋, j], Δτ, Δτλλ)
        Fˢᵢ[j] = reduce_linterm(Fˢₐ, λ, Fˢ[n₋, j], Δτ, Δτλλ)
    end

    return Fᵢ, Fˢᵢ
end

index_lin_svars(Λ, B, Bˢ, ΔF₁, ΔFˢ₁, Δτ, j) = (Λ[j], nothing, nothing, nothing, nothing, nothing)
index_lin_svars(Λ::Vector{<:TR}, B, Bˢ, ΔF₁, ΔFˢ₁, Δτ, j) = (Λ[j], B[j], Bˢ[j], ΔF₁[j], ΔFˢ₁[j], Δτ * Λ[j].t * Λ[j].r)

index_lin_tvars(Δζ, ζ, ζoζ::Nothing, Δτ, l, i) = (Δζ[l], nothing, nothing, nothing)
index_lin_tvars(Δζ, ζ, ζoζ, Δτ, l, i) = (Δζ[l], -Δτ * ζ[l], ζ[i].t * ζ[l].r, ζoζ[l - 1] - ζoζ[l])

initialize_linterm(F₁ⱼ, ζₙ, ζoζ::Nothing, n) = F₁ⱼ * ζₙ
#
function initialize_linterm(Fᵢⱼ, ζₙ, ζoζ, n)
    Fₐᵀ   = Fᵢⱼ.t * ζₙ.t  # Center-of-mass projection with coefficient λᵀ
    Fₐᴿᵀ  = Fᵢⱼ.r * ζₙ.t  # Projections with coefficient λᵀ
    Fₐᴿᴿ  = Fᵢⱼ.r * ζₙ.r  # Projections with coefficient λᴿ
    Fₐᴿᵀᴿ = Fᵢⱼ.r * ζoζ[n] # Projections with coefficient λᵀλᴿ

    return Fₐᵀ, Fₐᴿᵀ, Fₐᴿᴿ, Fₐᴿᵀᴿ
end

accumulate_linterm(Fₐ, Fᵢⱼ, Δζₗ, Δτζₗ::Nothing, ζᵢζₗ, Δζoζₗ, β, ΔF₁ⱼ) = muladd(Fᵢⱼ, Δζₗ, Fₐ)
#
function accumulate_linterm(Fₐ, Fᵢⱼ, Δζₗ, Δτζₗ, ζᵢζₗ, Δζoζₗ, β, ΔF₁ⱼ)
    Fₐᵀ, Fₐᴿᵀ, Fₐᴿᴿ, Fₐᴿᵀᴿ = Fₐ

    Fₐᵀ = muladd(Δζₗ.t , Fᵢⱼ.t, Fₐᵀ)
    Fₐᴿᵀ  = muladd.(muladd.(Δτζₗ.t, getr(β.r), Δζₗ.t), Fᵢⱼ.r, Fₐᴿᵀ)
    Fₐᴿᴿ  = muladd.(muladd.(Δτζₗ.r, getr(β.t), Δζₗ.r), Fᵢⱼ.r, Fₐᴿᴿ)
    Fₐᴿᵀᴿ = muladd.(Δζoζₗ, Fᵢⱼ.r, muladd.(ζᵢζₗ, ΔF₁ⱼ.r, Fₐᴿᵀᴿ))

    return Fₐᵀ, Fₐᴿᵀ, Fₐᴿᴿ, Fₐᴿᵀᴿ
end

reduce_linterm(Fₐ, λ, Fᵢⱼ, Δτ, Δτλλ::Nothing) = muladd(λ, Fₐ, Fᵢⱼ / Δτ)
#
function reduce_linterm(Fₐ, λ, Fᵢⱼ, Δτ, Δτλλ)
    Fₐᵀ, Fₐᴿᵀ, Fₐᴿᴿ, Fₐᴿᵀᴿ = Fₐ

    F̃ = Fᵢⱼ / Δτ
    t = muladd(λ.t, Fₐᵀ, F̃.t)
    r = muladd.(λ.t, Fₐᴿᵀ, muladd.(λ.r, Fₐᴿᴿ, muladd.(Δτλλ, Fₐᴿᵀᴿ, F̃.r)))

    return Projections.constructorname(F̃)(t, r)
end

## Nonlinear contributions of the SCGLE equations
function perform_nlstep!(F, Fˢ, Fᵢ, Fˢᵢ, ΔF₁, ΔFˢ₁, A, Aˢ, B, Bˢ, Z, ζ, Λ, w, υ, Δτ, i, tol)
    ζᵢ = ζ₁ = ζ[1]
    ζ[i] = init_guess(ζ, i)

    F₁ = view_firststep(F, 1)
    Fˢ₁ = view_firststep(Fˢ, 1)

    @inbounds while nonconvergent(ζᵢ, ζ[i], tol)
        ζᵢ = ζ[i]

        for j in eachindex(Λ)
            m, βm, βˢm = compute_nlmemory(Λ, ζᵢ, ζ₁, B, Bˢ, Δτ, j)
            F[i, j]  = reduce_nlterm(A[j], m, ΔF₁[j], Fᵢ[j], F₁, βm, j)
            Fˢ[i, j] = reduce_nlterm(Aˢ[j], m, ΔFˢ₁[j], Fˢᵢ[j], Fˢ₁, βˢm, j)
            Z[j] = product(w[j], F[i, j], Fˢ[i, j])
        end

        ζ[i] = υ * sum(Z)
    end
end

init_guess(ζ, i) = ζ[i - 1]

view_firststep(F, i) = nothing
view_firststep(F::Array{<:Projections.AbstractProjections}, i) = view(F, i, :)

nonconvergent(ζᵢ, ζ, tol) = abs(1 - ζᵢ / ζ) > tol
nonconvergent(ζᵢ::TR, ζ, tol) = (abs(1 - ζᵢ.t / ζ.t) > tol || abs(1 - ζᵢ.r / ζ.r) > tol)

compute_nlmemory(Λ, ζᵢ, ζ₁, B, Bˢ, Δτ, j) = (Λ[j] * ζᵢ, nothing, nothing)
#
function compute_nlmemory(Λ, ζᵢ::TR, ζ₁, B, Bˢ, Δτ, j)
    λ = Λ[j]

    m = λ * ζᵢ
    moᵢm = m.t * (1 + Δτ * λ.r * ζ₁.r) + m.r * (1 + Δτ * λ.t * ζ₁.t)
    βm  = -Δτ .* (getr(B[j].t) * m.r .+ getr(B[j].r) .* m.t)
    βˢm = -Δτ .* (Bˢ[j].t * m.r .+ Bˢ[j].r .* m.t)

    return (m.t, moᵢm), βm, βˢm
end

reduce_nlterm(α, m, ΔF₁ⱼ, Fᵢⱼ, F₁ⱼ, βm::Nothing, j) = α * muladd(m, ΔF₁ⱼ, Fᵢⱼ)
#
function reduce_nlterm(α, m, ΔF₁ⱼ, Fᵢⱼ, F₁, βm, j)
    mᵀ, moᵢm = m
    F₁ⱼ = F₁[j]

    t = α.t * muladd(mᵀ, ΔF₁ⱼ.t, Fᵢⱼ.t)
    r = α.r .* muladd.(moᵢm, ΔF₁ⱼ.r, muladd.(βm, F₁ⱼ.r, Fᵢⱼ.r))

    return Projections.constructorname(Fᵢⱼ)(t, r)
end
