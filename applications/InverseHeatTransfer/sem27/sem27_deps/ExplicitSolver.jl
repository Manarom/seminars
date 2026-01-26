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
    # allocating vectors of column size
    Fm =    Vector{Float64}(undef,N)
    phi_m = Vector{Float64}(undef,N)
    lam_m = Vector{Float64}(undef,N)
        for m = 1:M - 1 #% цикл по времени
            Tm = @view T[:,m]

            @inbounds @simd for ii in 1:N
                ti = Tm[ii]
                λ = L_f(ti) # λ
                lam_m[ii] = λ
                Fm[ii] = dd*λ/C_f(ti) # Fm - (dx^-2)*dt*Cp/λ
                phi_m[ii] = 0.25*Ld_f(ti)/λ #phi  - λ'/λ
            end 

            T[1, m + 1] = explicit_bc(UPPER_BC, upper_bc_type, Tm, BC_up_f, dt*(m - 1), Fm[1], phi_m[1],lam_m[1], dx)

            @inbounds for n = 2 : N - 1 #% цикл по координате
                r = Fm[n]
                T[n, m + 1] = explicit_iteration(Fm[n], phi_m[n], T[n - 1, m] , T[n , m], T[n + 1 , m])
                if  r > maxFn
                    maxFn = r;
                end
            end

            T[end, m + 1] = explicit_bc(LOWER_BC, lower_bc_type, Tm, BC_dwn_f, dt*(m - 1), Fm[end], phi_m[end],lam_m[end], dx)
        end
        return (T,dx,dt)
    end