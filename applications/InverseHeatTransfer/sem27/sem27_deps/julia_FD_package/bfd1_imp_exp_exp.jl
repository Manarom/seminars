const BFD1_IMP_EXP_EXP = FDSolverScheme{BFD1,IMP,EXP_NL,EXP_NL}
#=
            fill_LHS!(LHS, Fm1, F, Fp1, m, solver_scheme, problem)
            fill_RHS!(b, D, Tm, Tmm1, Fm1, F, Fp1, phi, solver_scheme, problem)
            apply_bc!(LHS, b, bc_fun,  F, Tm, phi ,  m,  solver_scheme, problem)
            ldiv!(Tmp1,LHS,b) # solving
=#
"""
LHS for `m+1`'th time step
`[... , -F, (1 + 2F),  - F , ...]T⁺ `
"""
function fill_LHS!(LHS, Fm1, F, Fp1, _ , ::BFD1_IMP_EXP_EXP, _ ) 
    fill_tridiag!(LHS, Fm1, F, Fp1, 1.0, -1.0, 2.0, -1.0)
end
"""
`T +  F*ϕ*(Tn+1 - Tn-1)² `
"""
function fill_RHS!(b, D, Tm, _ , _ , F, _ , phi, ::BFD1_IMP_EXP_EXP, _ ) 
            mul!(b,D,Tm) 
            @. b = b^2
            @. b *= F*phi
            @. b += Tm # Tm + \vec{b}
            return nothing
end

# Dirichlet BC

"""
    apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: DirichletBC , <: UpperBC}, _ , _ , m::Int , ::BFD1_IMP_EXP_EXP, _)   where {D,F,V}

`apply_bc!(LHS, b, bc_fun,  F, Tm, phi ,  m,  solver_scheme, problem)`

``[1, 0, ...] T⁺ = T⁺``

"""
function apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: DirichletBC , <: UpperBC}, _ , _ , m::Int , ::BFD1_IMP_EXP_EXP, _)   where {D , BF , V}
            LHS.d[1] = 1.0
            LHS.du[1] = 0.0
            b[1] = bc_fun(m + 1) # 1st order BC upper, evaluating bc for Tm+1
            return nothing
end
"""
    apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: DirichletBC , <: LowerBC}, _,_,  m::Int, ::BFD1_IMP_EXP_EXP, _)   where {D,F,V}

    `apply_bc!(LHS, b, bc_fun,  F, Tm, phi ,  m,  solver_scheme, problem)`
    ``` [..., 0 , 1] T⁺ = T⁺ ```
"""
function apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: DirichletBC , <: LowerBC}, _ , _ ,  m::Int, ::BFD1_IMP_EXP_EXP, _)   where {D , BF , V}
            LHS.d[end] = 1.0
            LHS.dl[end] = 0.0
            b[N] = bc_fun(m + 1) # 1st order BC upper, evaluating bc for Tm+1
            return nothing
end

"""
    apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: NeumanBC , <: UpperBC}, F , Tm ,  m::Int, ::BFD1_IMP_EXP_EXP, problem::HeatTransferProblem)   where {D,F,V}

    `apply_bc!(LHS, b, bc_fun,  F, Tm, phi ,  m,  solver_scheme, problem)`

    ```[(1 +  2F) , -2F , 0, ...] T⁺ =  T - 2F*Δx*q⁺/λ + 4F*ϕ*(Δx*q/λ)² ```

"""
function apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: NeumanBC , <: UpperBC}, F , Tm , phi,  m::Int, ::BFD1_IMP_EXP_EXP, problem::HeatTransferProblem)   where {D , BF , V}
            f = F[1]
            ϕ = phi[1]
            Ti = Tm[1]
            dx_div_λ = xstep(problem) / thermal_conductivity(problem, Ti)
            

            LHS.du[1] = - 2.0 * f
            LHS.d[1] = 1 + 2.0 * f

            
            qp1 = bc_fun(m + 1)
            q = bc_fun(m)
            b[1] = Tm[1] - 2.0 * dx_div_λ * qp1 + 4.0 * f * ϕ * (dx_div_λ * q)^2.0 
            return nothing
end

"""
    apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: NeumanBC , <: LowerBC}, F , Tm , phi,  m::Int, ::BFD1_IMP_EXP_EXP, problem::HeatTransferProblem)   where {D,F,V}

    `apply_bc!(LHS, b, bc_fun,  F, Tm, phi ,  m,  solver_scheme, problem)`

    ```[..., 0 , -2F , (1 +  2F)] T⁺ =  T - 2F*Δx*q⁺/λ + 4F*ϕ*(Δx*q/λ)² ```

"""
function apply_bc!(LHS, b,  bc_fun::BoundaryFunction{D, BF, V, <: NeumanBC , <: LowerBC}, F , Tm , phi,  m::Int, ::BFD1_IMP_EXP_EXP, problem::HeatTransferProblem)   where {D , BF , V}
            f = F[end]
            ϕ = phi[end]
            Ti = Tm[end]
            dx_div_λ = xstep(problem) / thermal_conductivity(problem, Ti)
            

            LHS.dl[end] = - 2.0 * f
            LHS.d[end] = 1 + 2.0 * f

            
            qp1 = bc_fun(m + 1)
            q = bc_fun(m)
            b[end] = Tm[end] - 2.0 * dx_div_λ * qp1 + 4.0 * f * ϕ * (dx_div_λ * q)^2.0 

            return nothing
end