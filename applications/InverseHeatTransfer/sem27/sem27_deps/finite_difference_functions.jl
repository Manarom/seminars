using LinearAlgebra
"""
    explicit_case1_dirichle(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

# C_f - функция теплоемкости от температуры(cp*Ro)
# L_f - функция теплопроводности от температуры
# Ld_f - функция производной теплопроводности от температуры
# H - толщина в м
# tmax - интервал времени
# initT_f - функция начального распределения температуры от координаты 
# BC_up_f - функция зависимости температуры сверху от времени
# BC_dwn_f - функция зависимости температуры снизу от времени
# N - число точек по координате
# M - число точек по времени
"""
function explicit_case1_dirichle(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

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
        return (T,x,t,maxFn)
    end
function allocate_tridiagonal(n::Int,T::DataType)
    dl = Vector{Float64}(undef, n-1)  # subdiagonal
    d  = Vector{Float64}(undef, n)    # diagonal  
    du = Vector{Float64}(undef, n-1)  # superdiagonal
    return Tridiagonal{}
end
function central_finite_difference(n)
    eu = fill(1.0, n-1)  # superdiagonal  
    el = fill(-1.0, n-1) # subdiagonal (first-order version)
    ed = fill(0.0, n)
    return  Tridiagonal(el,ed, eu)
end
"""
    implicit_case2_dirichle(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

# C_f - функция теплоемкости от температуры(cp*Ro)
# L_f - функция теплопроводности от температуры
# Ld_f - функция производной теплопроводности от температуры
# H - толщина в м
# tmax - интервал времени
# initT_f - функция начального распределения температуры от координаты 
# BC_up_f - функция зависимости температуры сверху от времени
# BC_dwn_f - функция зависимости температуры снизу от времени
# N - число точек по координате
# M - число точек по времени
"""
function implicit_case2_dirichle(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
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
        b = Vector{Float64}(undef,N)

        D = central_finite_difference(N) # creates finite difference matrix for b vector evaluation
        
        B0 = Vector{Float64}(undef, N) # B matrix diagonal
        Bm1 = Vector{Float64}(undef, N-1) # B matrix diagonal
        Bp1 = Vector{Float64}(undef, N-1) # B matrix diagonal

        B = Tridiagonal(Bm1,B0,Bp1) # creating tridiagonal matrix 

        Fmm1 =@view Fm[2 : end] 
        Fmp1 = @view Fm[1 : end - 1]

        for m = 1:M-1 #% цикл по времени
            Tm = @view T[:,m] # current time 
            @. lam_m = L_f(Tm) # теплопроводность для распределения температур  в m-й момент времени
            @. Fm = dd*lam_m/C_f(Tm) # Fm - число Фурье (dx^-2)*dt*Cp/lam
            @. phi_m = Ld_f(Tm)/(lam_m*4) #phi  - коэффициент при нелинейной функции
            Tmp1 = @view T[:,m + 1] # next time 
            # filling matrix diagonals
            @. B0 = 1 + 2*Fm
            @. Bm1 = - Fmm1
            @. Bp1 = - Fmp1
            # applying boundary conditions
            B0[1] = 1.0
            Bp1[1] = 0.0
            B0[end]= 1.0
            Bm1[end] =0.0    
            mul!(b,D,Tm)
            # filling RHS
            @. b = b^2
            @. b *= Fm*phi_m
            b[1] = Tmp1[1] - Tm[1]
            b[end] = Tmp1[end] - Tm[end]   
            @. b += Tm

            ldiv!(Tmp1,B,b)
        end
   return (T,x,t,maxFn)
end