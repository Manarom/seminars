
module OneDHeatTransfer

using LinearAlgebra,NonlinearSolvers, StaticArrays

using AllocCheck

const USE_FASTMATH = true
ni_err() = throw(DomainError("not implemented"))
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

const LOWER_BC = LowerBC()
const UPPER_BC = UpperBC()

const CentralDifference = Ref(Tridiagonal(fill(-1.0, 2),fill(0.0, 3), fill(1.0, 2))) # stores the finite difference matrix 

abstract type AbstractHTFunction{F,V} end
struct PhysicalPropertyFunction{F,V} <: AbstractHTFunction{F,V} 
    fun::F
    params::V
    function PhysicalPropertyFunction(f , p::V = nothing) where V
        fun = t -> f(t)
        F = typeof(fun)
        new{F,V}(fun,p)
    end 

end
struct BoundaryFunction{F, V, BCtype, BCdirection} <:AbstractHTFunction{F,V}
    fun::F
    value::V
    function BoundaryFunction(f , ::B , ::D, t) where {B  <: BCtype, D <: AbstractBCDirection} where BCtype <: Union{DirichletBC,NeumanBC}
        fun = t -> f(t)
        val = @. fun(t)
        V = typeof(val)
        F = typeof(fun)
        new{F,V,B,D}(fun,val)
    end 
    function BoundaryFunction(f, ::B , ::D, t) where { B <: BCtype, D <: AbstractBCDirection} where BCtype <: RobinBC
        val = nothing
        V = typeof(val)
        fun = t -> f(t)
        F = typeof(fun)
        new{F,V,B,D}(fun,val)
    end 
end

"""
    (bc::BoundaryFunction)(x)

Default implementation uses function call 
"""
(bc::AbstractHTFunction)(x) = bc.fun(x)

"""
    (bc::BoundaryFunction{F,BCtype} )(m::Int) where {F,BCtype <: Union{DirichletBC,NeumanBC}}

For DirichletBC and NeumanBC the value of BC can be evaluated in advance and collected by index
"""
(bc::BoundaryFunction{F,V,BCtype} )(m::Int) where {F,V,BCtype <: Union{DirichletBC,NeumanBC}} = bc.value[m]

include("AbstractGrid.jl")
include("TridiagFunctions.jl")

# Base.iterate(g::EachTimeStep) = 

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

include("ExplicitSolver.jl")

allocate_tridiagonal(N::Int,T::DataType = Float64) = Tridiagonal(Vector{T}(undef, N - 1),Vector{T}(undef, N),Vector{T}(undef, N - 1))

"""
    central_finite_difference(n)

Returns central finite difference first order derivative matrix 
"""
function central_finite_difference(n)
    if size(CentralDifference[],1) != n
         CentralDifference[] = Tridiagonal(fill(-1.0, n-1),fill(0.0, n), fill(1.0, n-1))   
    end  
    return  CentralDifference[]
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

"""
    apply_bc!(dir::AbstractBCDirection, bc_type::AbstractBoundaryCondition, 
                LHS, b,  F, bc_fun ,  
                time_scheme::AbstractTimeScheme, coord_scheme::AbstractCoordinateScheme,
                t, Tm, m, dx, dt, N)

# Arguments

    - dir - boundary direction (::LowerBC or ::UpperBC)
    - bc_type - type of boundary conditions (::DirichletBC,::NeumanBC,::RobinBC)
    - LHS - LHS matrix, can be modified
    - b - righthand side vector (next step after applying bc is mldivide(LHS,B))
    - F - vector of properties
    - bc_fun - boundary conditions function 
    - time_scheme - time scheme used for LHS (::BFD1, ::BFD2)
    - coord_scheme - scheme used for second derivative approximation 
    - t - current time 
    - Tm - current temperature distribution
    - m - current iteration
    - dx - coordinate step 
    - dt  - timestep
    - N - total number of coordinate points
# Retuns 


"""
function apply_bc! end

struct HeatTransferProblem{CF,LF,LDF,ITF,BUP,BDWN,G}
    C_f::CF 
    L_f::LF
    Ld_f::LDF
    initT_f::ITF
    g::G
end

function unified_fd_scheme( C_f, L_f,Ld_f, initT_f, 
                                g::UniformGrid{N,M,DType},
                                bc_up::BoundaryFunction, 
                                bc_dwn::BoundaryFunction,
                                time_scheme::AbstractTimeScheme,
                                coordinate_scheme::AbstractCoordinateScheme,
                                nln_scheme::AbstractNonLinearPart = EXP_NL(),
                                props_scheme::AbstractNonLinearPart = EXP_NL()
                ) where {N,M,DType}
        T = Matrix{DType}(undef,N,M)# columns - distribution, rows time
        T1 = @view T[:,1]
        map!(initT_f,T1,eachx(g))

        dd = g.dt/(g.dx*g.dx)#

        # allocating vectors and matrices
        F = Vector{DType}(undef,N) # properties vector
        Fm1 =@view F[2 : end] 
        Fp1 = @view F[1 : end - 1]
        phi = Vector{DType}(undef,N) # nonlinear coefficient vector λ'/λ
        lam = Vector{DType}(undef,N)
        b = Vector{DType}(undef,N) # left-hand part 
        D = central_finite_difference(N) # creates finite difference matrix for b vector evaluation
 
        Tm = T1

        # allocating left matrix
        LHS = allocate_tridiagonal(N) # left-hand side matrix 
        is_show = N <= 10
        for m = 1 : M - 1 #% цикл по времени
            Tm = @view T[:,m] # Tm current time
             @inbounds @simd for ii in 1:N
                ti = Tm[ii]
                λ = L_f(ti) # λ
                lam[ii] = λ
                F[ii] = dd*λ/C_f(ti) # Fm - (dx^-2)*dt*Cp/λ
                phi[ii] = 0.25*Ld_f(ti)/λ #phi  - λ'/λ
            end 
            Tmp1 = @view T[:, m + 1] # Tm+1 next time 
            Tmm1 = @view T[:, maximum((1, m - 1))] # Tm+1 next time 
            # filling matrix diagonals
            !is_show || @show m  
            fill_LHS!(LHS, Fm1, F, Fp1, time_scheme, coordinate_scheme, m)
            !is_show || @show LHS    
            fill_RHS!(b, D, Tm, Tmm1, Fm1, F, Fp1, phi, time_scheme, coordinate_scheme)
            !is_show || @show b
            !is_show || println("after bc")  
            apply_bc!(bc_up, LHS, b,  F,   time_scheme, coordinate_scheme, Tm, g, m)
            !is_show || @show LHS 
            apply_bc!(bc_dwn, LHS, b,  F,   time_scheme, coordinate_scheme, Tm, g, m)
            !is_show || @show b
            ldiv!(Tmp1,LHS,b) # solving
        end
   return T
end



fill_LHS!(LHS, Fm1, F, Fp1, ::BFD1, ::IMP, _) = fill_tridiag!(LHS,Fm1,F,Fp1,1.0,-1.0,2.0,-1.0)

"""
    fill_RHS!(b,RHS, D, Tm, Tmm1, Fm1, F, Fp1, phi, ::BFD1, ::IMP)

Fills RHS vector for BFD1 - IMP scheme
"""
function fill_RHS!(b, D, Tm, Tmm1, Fm1, F, Fp1, phi, ::BFD1, ::IMP)
            mul!(b,D,Tm) 
            @. b = b^2
            @. b *= F*phi
            @. b += Tm # Tm + \vec{b}
            return nothing
end

# apply_bc!(::AbstractBCDirection,upper_bc_type, LHS, b,  F, bc_fun ,  lhs_scheme, rhs_scheme, t, Tm, m, dx, dt, N)
function apply_bc!(f::BoundaryFunction{Ftype,V,BCtype,BCdirection}, 
            LHS::Tridiagonal{T,Vector{T}},
            b,  _,  
            ::BFD1, ::IMP, # time_scheme, coordinate_scheme
            Tm, # current timestep temperature distribution
            g::UniformGrid,
            m::Int  # current time step index
            ) where {T,Ftype,V, BCtype <: DirichletBC, BCdirection <: UpperBC}


            LHS.d[1] = 1.0
            LHS.du[1] = 0.0
            b[1] = f(m + 1) # 1st order BC upper, evaluating bc for Tm+1
            return nothing
end
function apply_bc!(f::BoundaryFunction{Ftype,V,BCtype,BCdirection}, 
            LHS::Tridiagonal{T,Vector{T}},
            b,  _,  
            ::BFD1, ::IMP, # time_scheme, coordinate_scheme
            Tm, # current timestep temperature distribution
            g::UniformGrid,
            m::Int  # current time step index
            ) where {T,Ftype,V, BCtype <: NeumanBC, BCdirection <: UpperBC}

            LHS.du[1] = 2.0*LHS.du[1]
            b[1] = f(m + 1) # 1st order BC upper, evaluating bc for Tm+1
            return nothing
end
function apply_bc!(f::BoundaryFunction{Ftype,V,BCtype,BCdirection}, 
            LHS::Tridiagonal{T,Vector{T}},
            b,  _,  
            ::BFD1, ::IMP, # time_scheme, coordinate_scheme
            Tm,
            g::UniformGrid{N},
            m::Int # current time step index
            ) where {T,N,Ftype,V, BCtype <: DirichletBC, BCdirection <: LowerBC}

            LHS.d[end] = 1.0
            LHS.dl[end] = 0.0
           
            b[N] = f(m + 1) # 1st order BC upper, evaluating bc for Tm+1
            return nothing
end

@doc DOC_BFD1_imp_exp_exp
 function BFD1_imp_exp_exp(C_f, L_f,Ld_f, H, tmax, initT_f,
                 bc_fun_up, bc_fun_dwn, M, N;
                 upper_bc_type::AbstractBoundaryCondition = DirichletBC() , 
                 lower_bc_type::AbstractBoundaryCondition = DirichletBC())

        g = UniformGrid(H,tmax,Val(N),Val(M))
        time_range = trange(g)

        bc_up = BoundaryFunction(bc_fun_up, upper_bc_type, UPPER_BC, time_range)
        bc_dwn = BoundaryFunction(bc_fun_dwn, lower_bc_type, LOWER_BC, time_range)

        T = unified_fd_scheme( C_f, L_f, Ld_f, initT_f, 
                                g,
                                bc_up, bc_dwn, 
                                BFD1(),
                                IMP())
        return (T,g,bc_up,bc_dwn)
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