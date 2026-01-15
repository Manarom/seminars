# runner for julia codes and benchmarking
using BenchmarkTools,Plots,Polynomials
plotly()
include("finite_difference_functions.jl")
function polyder(p)
    N = length(p)
    return ntuple(i->(N-i)*p[i],N-1)
end
lam_pars = (3e-7, 0, 0.442)
lam_poly = Polynomials.ImmutablePolynomial( reverse(lam_pars))
lam_fun = T -> lam_poly(T) # теплопроводность
lam_der_pars = polyder(lam_pars);
lam_der_poly = Polynomials.ImmutablePolynomial( reverse(lam_der_pars))
lam_der = T->lam_der_poly(T)#;% производная теплопроводности
#plot(linspace(200,1000,30),lam_fun(linspace(200,1000,30)));title("Теплопроводность, Вт/(м*К)")
#plot(linspace(200,1000,30),lam_der(linspace(200,1000,30)));title("Производная теплопроводности, Вт/(м*К^2)")

N = 50#;% число точек сетки по координате
M = 1000#;% число точек сетки по времени
Tmax = 1000.0#; % максимальная температура
tmax = 100.0#; % режим нагрева
Tinit = 20.0#; % начальная температура
Cp = 1000#; % теплоемкость
Ro = 2700#;% плотность
H = 15e-3#; % толщина слоя в мм
Cp_fun = _ -> Cp*Ro#;% не зависит от температуры
initT_f = _ -> Tinit #;% стартовая температура постоянна
BC_dwn_f = _ -> Tinit#;% температура снизу постоянна
BC_up_f = t -> Tinit + t*(Tmax - Tinit)/tmax#;% температура сверху линейно возрастает
#plot(linspace(0,tmax,100),BC_up_f(linspace(0,tmax,100)))##;title("Режим нагрева")
#% решаем диффур

(T,x,t,maxFn) = explicit_case1_dirichle(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

(T2,) = implicit_case2_dirichle(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
plot(T2,st=:surface)
#plot(T2 .-T,st=:surface)

(TCN,) = crank_nicolson_case3_dirichle(Cp_fun, lam_fun,lam_der, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)

plot(T2 .-TCN,st=:surface)