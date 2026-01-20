# runner for julia codes and benchmarking
using BenchmarkTools,Plots,Polynomials
#plotly()
using AllocCheck
include("finite_difference_functions.jl")

lam_pars = [0.44,0.01,1e-7]
lam_poly = Polynomials.ImmutablePolynomial(lam_pars)
lam_fun = Polynomials.ImmutablePolynomial(lam_pars) #T -> lam_poly(T) # теплопроводность
lam_der_poly = Polynomials.ImmutablePolynomial( derivative(lam_poly))
lam_der = Polynomials.ImmutablePolynomial( derivative(lam_poly))#T->lam_der_poly(T)#;% производная теплопроводности
plot(range(200.0,1000,30),lam_fun.(range(200.0,1000.0,30)))
plot(range(200.0,1000,30),lam_der.(range(200.0,1000.0,30)))
#plot(linspace(200,1000,30),lam_der(linspace(200,1000,30)));title("Производная теплопроводности, Вт/(м*К^2)")

N = 50#;% число точек сетки по координате
M = 5000#;% число точек сетки по времени
Tmax = 1000.0#; % максимальная температура
tmax = 100.0#; % режим нагрева
Tinit = 20.0#; % начальная температура
Cp = 1000.0#; % теплоемкость
Ro = 2700.0#;% плотность
H = 15e-3#; % толщина слоя в мм
@eval Cp_fun(_)= $Cp*$Ro#;% не зависит от температуры
@eval initT_f(_) =  $Tinit #;% стартовая температура постоянна
@eval BC_dwn_f(_) = $Tinit#;% температура снизу постоянна
@eval BC_up_f(t)=  $Tinit + t*($Tmax - $Tinit)/$tmax #;% температура сверху линейно возрастает

#plot(linspace(0,tmax,100),BC_up_f(linspace(0,tmax,100)))##;title("Режим нагрева")
#% решаем диффур

u_BC_type = OneDHeatTransfer.NeumanBC()
l_BC_type = OneDHeatTransfer.DirichletBC()


(T,) = OneDHeatTransfer.BFD1_exp_exp_exp(Cp_fun, lam_fun,lam_der, 
                        H, tmax,initT_f,
                        BC_up_f,BC_dwn_f,
                        M,N, upper_bc_type = u_BC_type)


plot(T,st=:surface)
OneDHeatTransfer.BFD1_exp_exp_exp(Cp_fun, lam_fun,lam_der, 
                        H, tmax,initT_f,
                        BC_up_f,BC_dwn_f,
                        M,N, upper_bc_type = u_BC_type)

(T2,) = OneDHeatTransfer.BFD1_imp_exp_exp(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
plot(T2,st=:surface)
#plot(T2 .-T,st=:surface)

(TCN,) = OneDHeatTransfer.BFD1_CN_exp_exp(Cp_fun, lam_fun,lam_der, 
            H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

plot(T2 .- TCN,st=:surface)

#@benchmark OneDHeatTransfer.BFD1_exp_exp_exp(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
@benchmark OneDHeatTransfer.BFD1_CN_exp_exp(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
@benchmark OneDHeatTransfer.BFD1_exp_exp_exp(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
@profview OneDHeatTransfer.BFD1_exp_exp_exp(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N, upper_bc_type = u_BC_type)