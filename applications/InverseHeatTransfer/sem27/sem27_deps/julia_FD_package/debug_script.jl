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
bc_fun = OD.BoundaryFunction(f_trial,collect(100.0:10.0:1000),OD.DirichletBC(),OD.UpperBC())
@btime bc_fun(1000.0)
@btime bc_fun(10)

bc_fun_rob = OD.BoundaryFunction(f_trial,collect(100.0:10.0:1000),OD.RobinBC(),OD.UpperBC())
@btime bc_fun_rob(1000.0)
bc_fun_rob(10)



struct callable_fun{F}
    f::F
end
(c::callable_fun)(x) =c.f(x) 

struct CallableFunAnnotated{F,D}
    f::F
    CallableFunAnnotated(f::F, ::Type{D}) where {F,D<:Number} = new{F,D}(f)  
end

((c::CallableFunAnnotated{F,D})(x)::D) where {F,D} = c.f(x)



using  BenchmarkTools
f_call = callable_fun(t -> t^2)
f_call_annot = CallableFunAnnotated(t -> t^2, Float64)
a = Vector{Float64}(undef,50)
@benchmark  a .= f_call.(a)
@benchmark a .= f_call_annot.(a)

@code_warntype f_call(2.5)
@code_warntype f_call_annot(3.5)

f2(a,f) =  @. f(a)

@code_warntype f2(a,f_call)

@benchmark f2(a,f_call)

@benchmark f2(a,f_call_annot)

@code_warntype f2(a,f_call)


module m1
    abstract type AbstractHTFunction{D, F, V} end
    function ((bc::AbstractHTFunction{D})(x)::D) where D
        bc.fun(x)
    end
    return_type(:: AbstractHTFunction{D}) where D = D
    struct PhysicalPropertyFunction{D,F,V} <: AbstractHTFunction{D,F,V} 
        fun::F
        params::V
        function PhysicalPropertyFunction(f , p::V = nothing, ::Type{D} = Float64) where {D <:Number,V}
            fun = t -> f(t)
            F = typeof(fun)
            new{D,F,V}(fun,p)
        end 
    end
end

module m2
    struct PhysicalPropertyFunction{D,F,V}
        fun::F
        params::V
        function PhysicalPropertyFunction(f , p::V = nothing, ::Type{D} = Float64) where {D <:Number,V}
            fun = t -> f(t)
            F = typeof(fun)
            new{D,F,V}(fun,p)
        end 
    end
    function ((bc::PhysicalPropertyFunction{D,F,V})(x)::D) where {D,F,V}
        bc.fun(x)
    end
end
module m3
    struct PhysicalPropertyFunction{D,F,V}
        fun::F
        params::V
        function PhysicalPropertyFunction(f , p::V = nothing, ::Type{D} = Float64) where {D <:Number,V}
            new{D,typeof(f),V}(f,p)
        end 
    end
    function ((bc::PhysicalPropertyFunction{D,F,V})(x)::D) where {D,F,V}
        bc.fun(x)
    end
end
f_1 = m1.PhysicalPropertyFunction(sin)
f_2 = m2.PhysicalPropertyFunction(sin)
f_3 = m3.PhysicalPropertyFunction(sin)
@benchmark f_1(45.8)
@benchmark f_2(45.8)
@benchmark f_3(45.8)


function view_test_fun!(V::Vector{T},A::Matrix{T}) where T
   # Tm = @view A[:,1]
    M = size(A,2)
    for m = 1 : M
        Tm = @view A[:,m]
        @. V .+= Tm 
    end
    return V
end
function view_copyto_test_fun!(V::Vector{T},A::Matrix{T}) where T
    Tm = @view A[:,1]
    M = size(A,2)
    for m = 1 : M
        copyto!(Tm, A[:,m])
        @. V .+= Tm 
    end
    return V
end
function optimal_view_test_fun!(V::Vector{T}, A::Matrix{T}) where T
    M = size(A, 2)
    views = [@view A[:,m] for m in 1:M]  # Single allocation!
    for m in 1:M
        @. V .+= views[m]
    end
    return V
end
function sum_columns!(V, A)
    @views colviews = [A[:,m] for m in 1:size(A,2)]
    @. V .= sum(colviews)  # Perfect fusion
end
A = rand(1000,1000)
V = zeros(Float64,(1000,))

@benchmark view_test_fun!($V,$A)
@benchmark view_copyto_test_fun!($V,$A)
@benchmark optimal_view_test_fun!($V,$A)
@benchmark sum_columns!($V,$A)