
module OneDHeatTransfer

using LinearAlgebra,NonlinearSolvers

using AllocCheck
 
abstract type AbstractLHS end
abstract type AbstractRHS end
abstract type AbstractNL end
abstract type AbstractBC end

struct DirichletBC<:AbstractBC end
struct NeumanBC<:AbstractBC end
struct RobinBC<: AbstractBC end

abstract type AbstractBCDirection end
struct LowerBC <: AbstractBCDirection end
struct UpperBC <: AbstractBCDirection end

struct BFD1_LHS <: AbstractLHS end
struct BFD2_LHS <: AbstractLHS end

struct exp_RHS <: AbstractRHS end
struct imp_RHS <: AbstractRHS end
struct CN_RHS <: AbstractRHS end

struct exp_NL <: AbstractNL end

const DDD = Ref(Tridiagonal(fill(-1.0, 2),fill(0.0, 3), fill(1.0, 2))) # stores the finite difference matrix 


"""
    fill_tridiag(Rm1,R0,Rp1,Fm1,F,Fp1,am1,a,ap1)

Fills three vectors with coefficients
"""
function fill_tridiag!(Rm1,R0,Rp1,Fm1,F,Fp1,a0,am1,a,ap1)
    @. Rm1 = am1*Fm1
    @. R0 = a0 + a*F
    @. Rp1 = ap1*Fp1
    return nothing
end


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
                 Tm, bc_fun, t, F, ϕ, λ, dx) where T<: AbstractBCDirection = explicit_hf_bc(bc_direction, Tm, bc_fun(t), F, ϕ, λ, dx)

explicit_bc(::UpperBC,::RobinBC,
                 Tm, bc_fun, t, F, ϕ, λ, dx) = explicit_hf_bc(UpperBC, Tm, bc_fun(Tm[1]), F, ϕ, λ, dx)

explicit_bc(::LowerBC,::RobinBC,
                 Tm, bc_fun, t, F, ϕ, λ, dx) = explicit_hf_bc(UpperBC, Tm, bc_fun(Tm[end]), F, ϕ, λ, dx)                 

function explicit_hf_bc(::UpperBC, Tm, hf, F, ϕ, λ, dx)
    T0 = Tm[2] - (2 * dx * hf / λ)                       
    return explicit_iteration(F, ϕ, T0, Tm[1], Tm[3])
end

function explicit_hf_bc(::LowerBC,Tm, hf, F, ϕ, λ, dx)
    TNp1 = Tm[end - 1] + (2 * dx * hf / λ)                       
    return explicit_iteration(F, ϕ,  Tm[end - 1], Tm[end], TNp1)
end
function explicit_iteration(F,fi,T1::T,T2::T,T3::T)
        return F * (T1 + T3) + (1 - 2*F) * T2 +  F * fi * (T1 - T3)^2 
end
#@doc DOC_BFD1_exp_exp_exp
@check_allocs function BFD1_exp_exp_exp(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N;
                 upper_bc_type::AbstractBC = DirichletBC() , 
                 lower_bc_type::AbstractBC = NeumanBC())

   #x = range(0,H,N)# сетка по координате
    #t = range(0,tmax,M)# сетка по времени
    dx = H/(N - 1)
    dt = tmax/(M - 1)
    T = Matrix{Float64}(undef,N,M)# columns - distribution, rows time
    T[:,1] .= initT_f.(0 : dx : H);
    #T[1,:] .= BC_up_f.(t)# applying upper BCs
    #T[N,:] .= BC_dwn_f.(t)#applying lower BC
    dd = dt/(dx*dx)#
    maxFn = 0.0;
    upper_bc_flag = UpperBC()
    lower_bc_flag = LowerBC()
    # allocating vectors of column size
    Fm = Vector{Float64}(undef,N)
    phi_m = Vector{Float64}(undef,N)
    lam_m = Vector{Float64}(undef,N)
        for m = 1:M-1 #% цикл по времени
            Tm = @view T[:,m]
            @. lam_m = L_f(Tm) # теплопроводность для распределения температур  в m-й момент времени
            @. Fm = dd*lam_m/C_f(Tm) # Fm - число Фурье (dx^-2)*dt*Cp/lam
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - коэффициент при нелинейной функции

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

function allocate_tridiagonal(N::Int,T::DataType = Float64)
    B0 = Vector{T}(undef, N) # B matrix diagonal
    Bm1 = Vector{T}(undef, N - 1) # B matrix diagonal
    Bp1 = Vector{T}(undef, N - 1) # B matrix diagonal

    B = Tridiagonal(Bm1,B0,Bp1) # creating tridiagonal matrix 
    return (B,Bm1,B0,Bp1)
end
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

fill_LHS!(dl,d0,du,Fm1,F,Fp1,::Type{BFD1_LHS},::Type{imp_RHS}) = fill_tridiag!(dl,d0,du,Fm1,F,Fp1,1.0,-1.0,2.0,-1.0)

@doc DOC_BFD1_imp_exp_exp
function BFD1_imp_exp_exp(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
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
        
        (L,Lm1,L0,Lp1) = allocate_tridiagonal(N)

        Fmm1 =@view Fm[2 : end] 
        Fmp1 = @view Fm[1 : end - 1]

        for m = 1:M-1 #% цикл по времени
            Tm = @view T[:,m] # Tm current time

            @. lam_m = L_f(Tm) # λ
            @. Fm = dd*lam_m/C_f(Tm) # Fm - (dx^-2)*dt*Cp/λ
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - λ'/λ

            Tmp1 = @view T[:,m + 1] # Tm+1 next time 
            # filling matrix diagonals
            fill_LHS!(Lm1, L0, Lp1, Fmm1, Fm, Fmp1, BFD1_LHS, imp_RHS)

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
        end
   return (T,x,t,maxFn)
end

fill_RHS!(dl,d0,du,Fm1,F,Fp1,::Type{BFD1_LHS},::Type{CN_RHS}) = fill_tridiag!(dl, d0, du, Fm1, F, Fp1, 1.0 , 0.5, 1.0, 0.5)

fill_LHS!(dl,d0,du,Fm1,F,Fp1,::Type{BFD1_LHS},::Type{CN_RHS}) = fill_tridiag!(dl, d0, du, Fm1, F, Fp1, 1.0, -0.5, 1.0, -0.5)

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

        for m = 1:M-1 #
            Tm = @view T[:,m] # Tm current time

            @. lam_m = L_f(Tm) # λ
            @. Fm = dd*lam_m/C_f(Tm) # Fm - (dx^-2)*dt*Cp/λ
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - λ'/λ

            Tmp1 = @view T[:,m + 1] # Tm+1 next time 
            # filling LHS matrix diagonals
            fill_LHS!(Lm1,L0,Lp1,Fmm1,Fm,Fmp1,BFD1_LHS,CN_RHS)

            # applying boundary conditions to the LHS
            L0[1] = 1.0
            Lp1[1] = 0.0
            L0[end]= 1.0
            Lm1[end] =0.0   
            
            # filling RHS matrix diagonals
            fill_RHS!(Rm1,R0,Rp1,Fmm1,Fm,Fmp1,BFD1_LHS,CN_RHS)

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