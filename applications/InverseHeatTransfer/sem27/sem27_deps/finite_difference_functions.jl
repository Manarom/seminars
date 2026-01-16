
module OneDHeatTransfer

using LinearAlgebra,NonlinearSolvers
 
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
# Customize signature per function in docstring if needed
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

@doc DOC_BFD1_exp_exp_exp
function BFD1_exp_exp_exp(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

    x = range(0,H,N)# сетка по координате
    t = range(0,tmax,M)# сетка по времени
    dx = x[2] - x[1]
    dt = t[2] - t[1]
    T = Matrix{Float64}(undef,N,M)# columns - distribution, rows time
    T[:,1] .= initT_f.(x);
    T[1,:] .= BC_up_f.(t)# applying upper BC
    T[N,:] .= BC_dwn_f.(t)#applying lower BC
    dd = dt/(dx*dx)#
    maxFn = 0.0;
    # allocating vectors of column size
    Fm = Vector{Float64}(undef,N)
    phi_m = Vector{Float64}(undef,N)
    lam_m = Vector{Float64}(undef,N)
        for m = 1:M-1 #% цикл по времени
            Tm = @view T[:,m]
            @. lam_m = L_f(Tm) # теплопроводность для распределения температур  в m-й момент времени
            @. Fm = dd*lam_m/C_f(Tm) # Fm - число Фурье (dx^-2)*dt*Cp/lam
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - коэффициент при нелинейной функции
            
            for n = 2:N-1 #% цикл по координате
                r = Fm[n];
                r1 = 1 - 2*r;
                fi = phi_m[n];
                b_n = r*fi*(T[n - 1,m] - T[n + 1,m])^2 # [\vec{b}^m(\vec{T}^m)]_n= F_{n}^m \phi_{n}^m (T^{m}_{n-1} - T^{m}_{n+1})^2
                T[n,m + 1] = r*(T[n - 1,m] + T[n + 1,m]) + r1*T[n,m] + b_n 
                if r > maxFn
                    maxFn = r;
                end
            end
        end
        return (T,x,t)
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
    eu = fill(1.0, n-1)  # superdiagonal  
    el = fill(-1.0, n-1) # subdiagonal (first-order version)
    ed = fill(0.0, n)
    return  Tridiagonal(el,ed, eu)
end

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
            @. L0 = 1 + 2*Fm # main diagonal
            @. Lm1 = - Fmm1 # lower diagonal
            @. Lp1 = - Fmp1 # upper diagonal
            # applying boundary conditions to the LHS
            L0[1] = 1.0
            Lp1[1] = 0.0
            L0[end]= 1.0
            Lm1[end] =0.0    
            # evaluating nonlinear term and righthand
            mul!(b,D,Tm) 
            @. b = b^2
            @. b *= Fm*phi_m
            b[1] = Tmp1[1] - Tm[1] # 1st order BC upper
            b[end] = Tmp1[end] - Tm[end]   # 1st order BC lower
            @. b += Tm # Tm + \vec{b}

            ldiv!(Tmp1,L,b) # solving
        end
   return (T,x,t,maxFn)
end

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
    bfd1_crank_nicolson_implicit_dirichle(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

Solves the finite-difference non-linear heat transfer using :

BFD1 for dT/dt

C_f - thermal capacity , (cp*Ro) (Kg/m^3 * J/(Kg*K))
L_f - thermal conductivity, W/m*K
Ld_f - thermal conductivity derivative with respect to temperature
H - thickness, m
tmax - time interval, s
initT_f - function to evaluate the initial temperature distribution
BC_up_f - upper BC function
BC_dwn_f - lower BC function 
N - points for coordinate
M - points for time
"""
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
end