include("finite_difference_functions.jl")
using Plots,BenchmarkTools,Cthulhu
import .OneDHeatTransfer as OD
N = 50
M = 150
g = OD.UniformGrid(15e-3,100.0,Val(N),Val(M))
a = Vector{Float64}(undef,N)
map!(sin,a,OD.eachx(g))
b = Vector{Float64}(undef,M)
bw = @view b[:]
map!(sin,bw,OD.eachtime(g))
plot(b)

@benchmark map!(sin,bw,OD.eachtime(g))
@code_warntype map!(sin,bw,OD.eachtime(g))

f_trial = t -> t/1000.0
bc_fun = OD.BoundaryFunction(f_trial,OD.DirichletBC(),OD.UpperBC(),collect(100.0:10.0:1000))
@btime bc_fun(1000.0)
@btime bc_fun(10)

bc_fun_rob = OD.BoundaryFunction(f_trial,OD.RobinBC(),OD.UpperBC(),collect(100.0:10.0:1000))
@btime bc_fun_rob(1000.0)
bc_fun_rob(10)