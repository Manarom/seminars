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