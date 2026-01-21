
module OneDHeatTransfer

using LinearAlgebra,NonlinearSolvers

using AllocCheck
# this types are used to dispatch on different schemes  
abstract type AbstractTimeScheme end # left-hand side part 
abstract type AbstractCoordinateScheme end # right-hand side part 
abstract type AbstractNonLinearPart end # non-linear part type

# singletones for various schemes
struct BFD1 <: AbstractTimeScheme end # first order backward difference
struct BFD2 <: AbstractTimeScheme end # second order backward difference

struct EXP <: AbstractCoordinateScheme end # fully explicit 
struct IMP <: AbstractCoordinateScheme end # fully implicit 
struct CN <: AbstractCoordinateScheme end # crank - nicolson

struct EXP_NL <: AbstractNonLinearPart end # explicit non - linear part
struct IMP_NL <: AbstractNonLinearPart end # implicit non - linear part


# boundaries conditions types 
abstract type AbstractBoundaryCondition end
struct DirichletBC <: AbstractBoundaryCondition end
struct NeumanBC <: AbstractBoundaryCondition end
struct RobinBC <: AbstractBoundaryCondition end

# boundary type upper or lower 
abstract type AbstractBCDirection end
struct LowerBC <: AbstractBCDirection end
struct UpperBC <: AbstractBCDirection end

const DDD = Ref(Tridiagonal(fill(-1.0, 2),fill(0.0, 3), fill(1.0, 2))) # stores the finite difference matrix 


"""
    fill_tridiag!(M::Tridiagonal{T,Vector{T}},Fm1::V,F::V,Fp1::V,a0,am1,a,ap1) where V <: AbstractVector{T} where T


Fills three vectors with coefficients
"""
function fill_tridiag!(M::Tridiagonal{T,Vector{T}},
                                    Fm1::AbstractVector{T} ,F::AbstractVector{T} ,
                                    Fp1::AbstractVector{T} ,a0,am1,a,ap1)  where T

    @. M.dl = am1*Fm1
    @. M.d = a0 + a*F
    @. M.du = ap1*Fp1
    return nothing
end
function fill_tridiag!(M::Tridiagonal{T,Vector{T}},Fm1::AbstractVector{T},F::AbstractVector{T},Fp1::AbstractVector{T},a0,am1,a,ap1,k::Int)  where T
    M.dl[k] = am1*Fm1[k]
    M.d[k] = a0 + a*F[k]
    M.du[k] = ap1*Fp1[k]
    return nothing
end
function tridiag_sum(Fm1::AbstractVector{T},F::AbstractVector{T}, Fp1::AbstractVector{T}, a0, am1, a, ap1, k::Int) where T
    return a0 + am1*Fm1[k] +  a*F[k] + ap1*Fp1[k]
end

"""
    tridiag_vect_prod!(b::AbstractVector{T},Fm1::AbstractVector{T},
                                    F::AbstractVector{T}, Fp1::AbstractVector{T}, a0, 
                                    am1, a, ap1,N::Int)  where T

fills the vector b = M*b as M = Tridiagonal(Fm1*am1, a0 + a*F, Fp1*ap1) , N is the length of b

"""
function tridiag_mul!(b::AbstractVector{T},Fm1::AbstractVector{T},
                                    F::AbstractVector{T}, Fp1::AbstractVector{T}, a0, 
                                    am1, a, ap1,N::Int)  where T

    (N == length(b) && N == length(F) && N - 1 == length(Fp1) && N - 1 == length(Fm1)) || throw(DimensionMismatch("incosistent dimentions"))

    bm1 = b[1]

    b[1] = (a0 + a*F[1])*b[1] + ap1*Fp1[1]*b[2]
    @inbounds for ii in 2 : N - 1
        b1, b2, b3 = bm1, b[ii], b[ii + 1]
        f1,f2,f3 = Fm1[ii - 1],F[ii],Fp1[ii] 
        b[ii] = am1*b1*f1+ b2*(a0 + a*f2) + ap1*b3*f3
        bm1 = b2
    end
    b[N] = b[N]*(a0 + a*F[N]) + am1*bm1*Fm1[N - 1] 

    return nothing
end

function column_sym_tridiag_mul!(b::AbstractVector{T}, F::AbstractVector{T}, a0, a, ap1, N::Int)  where T

    (N == length(b) && N == length(F) ) || throw(DimensionMismatch("incosistent dimentions"))

    bm1 = b[1]

    b[1] = (a0 + a*F[1])*b[1] + ap1*F[1]*b[2]

    @inbounds for ii in 2 : N - 1
        b1, b2, b3 = bm1, b[ii], b[ii + 1]
        f = F[ii]
        b[ii] = 2*ap1*(b1 + b3)*f + b2*(a0 + a*f)
        bm1 = b2
    end

    b[N] = b[N] * ( a0 + a*F[N] ) + ap1 * bm1 * Fm1[N - 1] 

    return nothing
end


"""
    tridiag_vect_prod!(b::AbstractVector{T},Fm1::AbstractVector{T},
                                    F::AbstractVector{T}, Fp1::AbstractVector{T}, a0, 
                                    am1, a, ap1,N::Int)  where T

fills the vector b = M*c + b as M = Tridiagonal(Fm1*am1, a0 + a*F, Fp1*ap1) , N is the length of b

"""
function tridiag_muladd!(b::AbstractVector{T},c::AbstractVector{T}, Fm1::AbstractVector{T},
                                    F::AbstractVector{T}, Fp1::AbstractVector{T}, a0, 
                                    am1, a, ap1,N::Int)  where T

    (N==length(c) && N == length(b) && N == length(F) && N - 1 == length(Fp1) && N - 1 == length(Fm1)) || throw(DimensionMismatch("incosistent dimentions"))


    b[1] = (a0 + a*F[1])*c[1] + ap1 * Fp1[1]*c[2]

    @inbounds for ii in 2 : N - 1
        c1, c2, c3 = c[ii - 1], c[ii], c[ii + 1]
        f1, f2, f3 = Fm1[ii - 1], F[ii], Fp1[ii] 
        b[ii] += am1*c1*f1+ c2*(a0 + a*f2) + ap1*c3*f3
    end

    b[N] = c[N]*(a0 + a*F[N]) + am1*bm1*Fm1[N - 1] 

    return nothing
end

tridiag_sum(Fm1, F, Fp1, a0, am1, a, ap1) = a0 + ap1*Fp1  +  a*F + am1*Fm1

const COMMON_DOC = """

    func(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

    Function solves the one dimentional heat transfer equation:

    (C/λ)*Tₜ= Tₓₓ + (λ'/λ)*(Tₓ)²
    T(0,t) = f(t)
    T(H,t) = g(t)
    T(x,0) = Tᵢ(x)

    where
    λ - thermal conductivity, Kg/m^3 * J/(Kg*K)    
    C - thermal capacity,    C = Cp*ρ    Cp - specific heat, J/(kg*K), ρ - density, kg/m³  
    Tₜ = ∂T/∂t  
    Tₓ = ∂T/∂x
    Tₓₓ = ∂²T/∂x²     

    with SCHEME_NAME finite difference scheme

    # Arguments

    - C_f - thermal capacity , (cp*Ro) (Kg/m^3 * J/(Kg*K))
    - L_f - thermal conductivity, W/m*K
    - Ld_f - thermal conductivity derivative with respect to temperature
    - H - thickness, m
    - tmax - time interval, s
    - initT_f - function to evaluate the initial temperature distribution
    - BC_up_f - upper BC function
    - BC_dwn_f - lower BC function 
    - N - points for coordinate
    - M - points for time

    # Returns

    (T,x,t) - (temperature matrix each column - distribution over coordinate, coordinates, time vector)

"""

const func_names = [:BFD1_exp_exp_exp,
              :BFD1_imp_exp_exp,
              :BFD1_CN_exp_exp]

export BFD1_exp_exp_exp,BFD1_imp_exp_exp,BFD1_CN_exp_exp
# generating docstrings
for d in func_names
    sd = string(d)
    full_name = replace(sd,"BFD1" => "first-order-backward =",
                            "exp" => "explicit",
                            "imp" => "implicit",
                            "CN" => "Crank-Nicolson",
                            "BFD2" =>"second-order-backward =",
                            "_" => " + ")
    dsds = Symbol("DOC_"*sd)
    cur_doc = replace(COMMON_DOC,"SCHEME_NAME" => full_name)
    @eval $dsds = replace($cur_doc,"func" => $sd)
end

explicit_bc(::T,::DirichletBC, Tm, bc_fun, t, F, ϕ, λ, dx) where T<: AbstractBCDirection= bc_fun(t)

explicit_bc(bc_direction::T,::NeumanBC,
                 Tm, bc_fun, t, F, ϕ, λ, dx) where T <: AbstractBCDirection = explicit_hf_bc(bc_direction, Tm, bc_fun(t), F, ϕ, λ, dx)

explicit_bc(bc::UpperBC,::RobinBC,
                 Tm, bc_fun, t, F, ϕ, λ, dx) = explicit_hf_bc(bc, Tm, bc_fun(Tm[1]), F, ϕ, λ, dx)

explicit_bc(bc::LowerBC,::RobinBC,
                 Tm, bc_fun, t, F, ϕ, λ, dx) = explicit_hf_bc(bc, Tm, bc_fun(Tm[end]), F, ϕ, λ, dx)                 

function explicit_hf_bc(::UpperBC, Tm, hf, F, ϕ, λ, dx)
    T0 = Tm[2] + (2 * dx * hf / λ)                       
    return explicit_iteration(F, ϕ, T0, Tm[1], Tm[3])
end

function explicit_hf_bc(::LowerBC,Tm, hf, F, ϕ, λ, dx)
    TNp1 = Tm[end - 1] + (2 * dx * hf / λ)                       
    return explicit_iteration(F, ϕ,  Tm[end - 1], Tm[end], TNp1)
end
function explicit_iteration(F,fi,T1,T2,T3)
        return F * (T1 + T3) + (1 - 2*F) * T2 +  F * fi * (T1 - T3)^2 
end
#@doc DOC_BFD1_exp_exp_exp
 function BFD1_exp_exp_exp(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N;
                 upper_bc_type::AbstractBoundaryCondition = DirichletBC() , 
                 lower_bc_type::AbstractBoundaryCondition = NeumanBC())

   #x = range(0,H,N)# сетка по координате
    #t = range(0,tmax,M)# сетка по времени
    dx = H/(N - 1)
    dt = tmax/(M - 1)
    T = Matrix{Float64}(undef,N,M)# columns - distribution, rows time
    T1 = @view T[:,1]
    @. T1 = initT_f(0 : dx : H)
    #T[1,:] .= BC_up_f.(t)# applying upper BCs
    #T[N,:] .= BC_dwn_f.(t)#applying lower BC
    dd = dt/(dx*dx)#
    maxFn = 0.0;
    upper_bc_flag = UpperBC()
    lower_bc_flag = LowerBC()
    # allocating vectors of column size
    Fm =    Vector{Float64}(undef,N)
    phi_m = Vector{Float64}(undef,N)
    lam_m = Vector{Float64}(undef,N)
    C_m =   Vector{Float64}(undef,N)
        for m = 1:M - 1 #% цикл по времени
            Tm = @view T[:,m]

            @. lam_m = L_f(Tm) # теплопроводность для распределения температур  в m-й момент времени
            @. C_m = C_f(Tm)
            @. Fm = dd*lam_m/C_m # Fm - число Фурье (dx^-2)*dt*Cp/lam
            @. phi_m = 0.25 * Ld_f(Tm) #phi  - коэффициент при нелинейной функции
            @. phi_m /= lam_m

            T[1, m + 1] = explicit_bc(upper_bc_flag, upper_bc_type, Tm, BC_up_f, dt*(m - 1), Fm[1], phi_m[1],lam_m[1], dx)

            for n = 2 : N - 1 #% цикл по координате
                r = Fm[n]
                T[n, m + 1] = explicit_iteration(Fm[n], phi_m[n], T[n - 1, m] , T[n , m], T[n + 1 , m])
                if  r > maxFn
                    maxFn = r;
                end
            end

            T[end, m + 1] = explicit_bc(lower_bc_flag, lower_bc_type, Tm, BC_dwn_f, dt*(m - 1), Fm[end], phi_m[end],lam_m[end], dx)
        end
        return (T,dx,dt)
    end

allocate_tridiagonal(N::Int,T::DataType = Float64) = Tridiagonal(Vector{T}(undef, N - 1),Vector{T}(undef, N),Vector{T}(undef, N - 1))

"""
    central_finite_difference(n)

Returns central finite difference first order derivative matrix 
"""
function central_finite_difference(n)
    if size(DDD[],1) != n
         DDD[] = Tridiagonal(fill(-1.0, n-1),fill(0.0, n), fill(1.0, n-1))   
    end  
    return  DDD[]
end


"""
    fill_LHS!(LHS, Fm1, F, Fp1, lhs_type, rhs_type)

Fills left-hand side of f-d scheme equation

LHS - matrix to be filled 
Fm1 - Fm-1 lower diagonal fourier number vector
F -   Fm  - main diagonal fourier number vector
Fp1 - Fm+1 upper diagonal fourier number vector 
lhs_type - left-hand side f-d scheme
rhs_type - right-hand side f-d scheme
m - iteration number
"""
function fill_LHS!(LHS, Fm1, F, Fp1, lhs_type, rhs_type, m) error("This scheme is not implemented yet") end


"""
    fill_RHS!(b, D, Tm, Tmm1, Fm1, F, Fp1, phi, lhs_type, rhs_type)

b - left-hand side vector to be filled (modified)
D - finite difference matrix 
Tm - temperature distribution for m'th timestep
Tmm1 - m-1 th timestep temperature distribution
Fm1, F, Fp1 - are  Fm-1 , Fm , Fm+1 elements of fourier number vector (views of the same vector)
phi - nonlinear coefficients vector 
time_scheme - type of time scheme 
coordinate_scheme - coordinate scheme 

"""
function fill_RHS!(b, D, Tm, Tmm1, Fm1, F, Fp1, phi, 
    time_scheme::AbstractTimeScheme,
     coordinate_scheme::AbstractCoordinateScheme) error("This scheme is not implemented yet") end

function apply_bc!(direction::AbstractBCDirection)

end

function unified_fd_scheme( C_f, L_f,Ld_f, H, tmax, initT_f, BC_up_f, BC_dwn_f, M, N,
                                upper_bc_type::AbstractBoundaryCondition, 
                                lower_bc_type::AbstractBoundaryCondition,
                                lhs_scheme::AbstractTimeScheme,
                                rhs_scheme::AbstractCoordinateScheme,
                                nln_scheme::AbstractNonLinearPart = EXP_NL(),
                                props_scheme::AbstractNonLinearPart = EXP_NL()
                )
        dx = H/(N - 1)
        dt = tmax/(M - 1)
        T = Matrix{Float64}(undef,N,M)# columns - distribution, rows time
        Tmm1 = @view T[:,1]
        @. T1 = initT_f(0 : dx : H)

        dd = dt/(dx*dx)#

        # allocating vectors and matrices
        F = Vector{Float64}(undef,N) # properties vector
        Fm1 =@view Fm[2 : end] 
        Fp1 = @view Fm[1 : end - 1]
        phi = Vector{Float64}(undef,N) # nonlinear coefficient vector λ'/λ
        lam = Vector{Float64}(undef,N)
        b = Vector{Float64}(undef,N) # left-hand part 
        D = central_finite_difference(N) # creates finite difference matrix for b vector evaluation
        
        # allocating left and right matrices
        LHS = allocate_tridiagonal(N) # left-hand side matrix 

        for m = 1 : M-1 #% цикл по времени
            Tm = @view T[:,m] # Tm current time
            t = dt*(m - 1)
            @. lam = L_f(Tm) # λ
            @. F = dd*lam/C_f(Tm) # Fm - (dx^-2)*dt*Cp/λ
            @. phi = 0.25*Ld_f(Tm)/lam #phi  - λ'/λ

            Tmp1 = @view T[:,m + 1] # Tm+1 next time 
            # filling matrix diagonals
            fill_LHS!(LHS, Fm1, F, Fp1, lhs_type, rhs_type, m)
            fill_RHS!(b,RHS, D, Tm, Tmm1, Fm1, F, Fp1, phi, lhs_scheme, rhs_scheme)
            apply_upper_bc!(upper_bc_type, LHS, b,  F,  b, lhs_scheme, rhs_scheme, t, dx, dt,N)
            apply_lower_bc!(lower_bc_type, LHS, b,  F,  b, lhs_scheme, rhs_scheme, t, dx, dt,N)

            # applying boundary conditions to the LHS
            #=L0[1] = 1.0
            Lp1[1] = 0.0
            L0[end]= 1.0
            Lm1[end] =0.0    
            # evaluating nonlinear term and righthand
            mul!(b,D,Tm) 
            @. b = b^2
            @. b *= Fm*phi_m
            @. b += Tm # Tm + \vec{b}

            b[1] = Tmp1[1]  # 1st order BC upper
            b[end] = Tmp1[end]   # 1st order BC lower
            =#

            ldiv!(Tmp1,LHS,b) # solving
        end
   return (T,dx,dt)
end



fill_LHS!(LHS, Fm1, F, Fp1, ::BFD1, ::IMP, _) = fill_tridiag!(LHS,Fm1,F,Fp1,1.0,-1.0,2.0,-1.0)
"""
    fill_RHS!(b,RHS, D, Tm, Tmm1, Fm1, F, Fp1, phi, ::BFD1, ::IMP)

Fills RHS vector for BFD1 - IMP scheme
"""
function fill_RHS!(b,_, D, Tm, Tmm1, Fm1, F, Fp1, phi, ::BFD1, ::IMP)
            mul!(b,D,Tm) 
            @. b = b^2
            @. b *= F*phi
            @. b += Tm # Tm + \vec{b}
            return nothing
end
@doc DOC_BFD1_imp_exp_exp
function BFD1_imp_exp_exp(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N;
                 upper_bc_type::AbstractBoundaryCondition = DirichletBC() , 
                 lower_bc_type::AbstractBoundaryCondition = NeumanBC())

        #=x = range(0,H,N)# сетка по координате
        t = range(0,tmax,M)# сетка по времени
        dx = x[2] - x[1]
        dt = t[2] - t[1]
        T = Matrix{Float64}(undef,N,M)# columns - distribution, rows time
        T[:,1] .= initT_f.(x)# applying initial conditions
        T[1,:] .= BC_up_f.(t)# applying upper BC
        T[N,:] .= BC_dwn_f.(t)#applying lower BC
        dd = dt/(dx*dx)#
        maxFn = 0.0;
        # allocating vectors of column size
        Fm = Vector{Float64}(undef,N)
        phi_m = Vector{Float64}(undef,N)
        lam_m = Vector{Float64}(undef,N)
        b = Vector{Float64}(undef,N) =#

        return unified_fd_scheme( C_f, L_f,Ld_f, H, tmax, initT_f, BC_up_f, BC_dwn_f, M, N,
                                upper_bc_type::AbstractBoundaryCondition, 
                                lower_bc_type::AbstractBoundaryCondition,
                                 BFD1(),
                                IMP())
        #=
        D = central_finite_difference(N) # creates finite difference matrix for b vector evaluation
        
        # allocating left and right matrices
        
        L = allocate_tridiagonal(N)

        Fmm1 =@view Fm[2 : end] 
        Fmp1 = @view Fm[1 : end - 1]

        for m = 1:M-1 #% цикл по времени
            Tm = @view T[:,m] # Tm current time

            @. lam_m = L_f(Tm) # λ
            @. Fm = dd*lam_m/C_f(Tm) # Fm - (dx^-2)*dt*Cp/λ
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - λ'/λ

            Tmp1 = @view T[:,m + 1] # Tm+1 next time 
            # filling matrix diagonals
            fill_LHS!(L, Fmm1, Fm, Fmp1, lhs_type, rhs_type)

            # applying boundary conditions to the LHS
            L0[1] = 1.0
            Lp1[1] = 0.0
            L0[end]= 1.0
            Lm1[end] =0.0    
            # evaluating nonlinear term and righthand
            mul!(b,D,Tm) 
            @. b = b^2
            @. b *= Fm*phi_m
            @. b += Tm # Tm + \vec{b}

            b[1] = Tmp1[1]  # 1st order BC upper
            b[end] = Tmp1[end]   # 1st order BC lower


            ldiv!(Tmp1,L,b) # solving
        end =#
   return (T,x,t,maxFn)
end

fill_RHS!(M,Fm1,F,Fp1,::BFD1,::CN) = fill_tridiag!(M, Fm1, F, Fp1, 1.0 , 0.5, 1.0, 0.5)

fill_LHS!(M,Fm1,F,Fp1,::BFD1,::CN) = fill_tridiag!(M, Fm1, F, Fp1, 1.0, -0.5, 1.0, -0.5)

@doc DOC_BFD1_CN_exp_exp
function BFD1_CN_exp_exp(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
        x = range(0,H,N)# сетка по координате
        t = range(0,tmax,M)# сетка по времени
        dx = x[2] - x[1]
        dt = t[2] - t[1]
        T = Matrix{Float64}(undef,N,M)# columns - distribution, rows time
        T[:,1] .= initT_f.(x)# applying initial conditions
        T[1,:] .= BC_up_f.(t)# applying upper BC
        T[N,:] .= BC_dwn_f.(t)#applying lower BC
        dd = dt/(dx*dx)#
        maxFn = 0.0;
        # allocating vectors of column size
        Fm = Vector{Float64}(undef,N)
        phi_m = Vector{Float64}(undef,N)
        lam_m = Vector{Float64}(undef,N)
        b = Vector{Float64}(undef,N)

        D = central_finite_difference(N) # creates finite difference matrix for b vector evaluation
        
        # allocating left and right matrices
        (R,Rm1,R0,Rp1) = allocate_tridiagonal(N)
        (L,Lm1,L0,Lp1) = allocate_tridiagonal(N)

        Fmm1 =@view Fm[2 : end] 
        Fmp1 = @view Fm[1 : end - 1]
        lhs_obj = BFD1()
        rhs_obj = CN()
        for m = 1:M-1 #
            Tm = @view T[:,m] # Tm current time

            @. lam_m = L_f(Tm) # λ
            @. Fm = dd*lam_m/C_f(Tm) # Fm - (dx^-2)*dt*Cp/λ
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - λ'/λ

            Tmp1 = @view T[:,m + 1] # Tm+1 next time 
            # filling LHS matrix diagonals
            fill_LHS!(Lm1,L0,Lp1,Fmm1,Fm,Fmp1,lhs_obj,rhs_obj)

            # applying boundary conditions to the LHS
            L0[1] = 1.0
            Lp1[1] = 0.0
            L0[end]= 1.0
            Lm1[end] =0.0   
            
            # filling RHS matrix diagonals
            fill_RHS!(Rm1,R0,Rp1,Fmm1,Fm,Fmp1,lhs_obj,rhs_obj)

            R0[1] = 1.0
            Rp1[1] = 0.0
            R0[end]= 1.0
            Rm1[end] =0.0   

            # filling RHS
            mul!(b,D,Tm)
            @. b = b^2
            @. b *= Fm*phi_m
            b[1] = Tmp1[1] - Tm[1] # 1st order BC upper
            b[end] = Tmp1[end] - Tm[end]   # 1st order BC lower

            mul!(b, R, Tm, 1.0, 1.0) # b = b + R*Tm
            #@. b += R*Tm # Tm + \vec{b}

            ldiv!(Tmp1,L,b)
        end
   return (T,x,t,maxFn)
end

function BFD2_CN_exp_exp(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
    error("TODO")
        x = range(0,H,N)# сетка по координате
        t = range(0,tmax,M)# сетка по времени
        dx = x[2] - x[1]
        dt = t[2] - t[1]
        T = Matrix{Float64}(undef,N,M)# columns - distribution, rows time
        T[:,1] .= initT_f.(x)# applying initial conditions
        T[1,:] .= BC_up_f.(t)# applying upper BC
        T[N,:] .= BC_dwn_f.(t)#applying lower BC
        dd = dt/(dx*dx)#
        maxFn = 0.0;
        # allocating vectors of column size
        Fm = Vector{Float64}(undef,N)
        phi_m = Vector{Float64}(undef,N)
        lam_m = Vector{Float64}(undef,N)
        b = Vector{Float64}(undef,N)

        D = central_finite_difference(N) # creates finite difference matrix for b vector evaluation
        
        # allocating left and right matrices
        (R,Rm1,R0,Rp1) = allocate_tridiagonal(N)
        (L,Lm1,L0,Lp1) = allocate_tridiagonal(N)

        Fmm1 =@view Fm[2 : end] 
        Fmp1 = @view Fm[1 : end - 1]

        for m = 1:M-1 #% цикл по времени
            Tm = @view T[:,m] # Tm current time
            @. lam_m = L_f(Tm) # λ
            @. Fm = dd*lam_m/C_f(Tm) # Fm - (dx^-2)*dt*Cp/λ
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - λ'/λ
            Tmp1 = @view T[:,m + 1] # Tm+1 next time 
            # filling LHS matrix diagonals
            @. L0 = 1 + Fm
            @. Lm1 = - Fmm1/2
            @. Lp1 = - Fmp1/2
            # applying boundary conditions to the LHS
            L0[1] = 1.0
            Lp1[1] = 0.0
            L0[end]= 1.0
            Lm1[end] =0.0   
            
            # filling RHS matrix
            @. R0 = 1 - Fm
            @. Rm1 = Fmm1/2
            @. Rp1 = Fmp1/2
            R0[1] = 1.0
            Rp1[1] = 0.0
            R0[end]= 1.0
            Rm1[end] =0.0   


            # filling RHS
            mul!(b,D,Tm)
            @. b = b^2
            @. b *= Fm*phi_m
            b[1] = Tmp1[1] - Tm[1] # 1st order BC upper
            b[end] = Tmp1[end] - Tm[end]   # 1st order BC lower

            mul!(b, R, Tm, 1.0, 1.0) # b = b + R*Tm
            #@. b += R*Tm # Tm + \vec{b}

            ldiv!(Tmp1,L,b)
        end
   return (T,x,t,maxFn)
end

"""
Bunch of functions to solve the non-linear transient heat transfer using finite difference

    a(T)Tₜ= Tₓₓ + (λ'/λ)*(Tₓ)²
    T(0,t) = f(t)
    T(H,t) = g(t)
    T(x,0) = Tᵢ(x)

    Name convention of the functions:

    lefthand side  _ second derivative _ nonlinear part _ material properties
    e.g.:
    BFD1_imp_exp_exp  => first order time derivative,
    implicit scheme for the second derivative, explicit nonlinear part, explicit physical properties

    imp - fully implicit
    exp - fully explicit
    CN - Crank-Nicolson
    BFD1 - first order backward derivative
    BFD2 - second order backward werivative

"""
OneDHeatTransfer


end