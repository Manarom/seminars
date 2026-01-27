
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
abstract type AbstractHTFunction{D, F, V} end
"""
    (bc::AbstractHTFunction)(x)

Default implementation uses function call 
"""
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

struct InitialTFunction{D, F, V} <: AbstractHTFunction{D, F, V} 
    fun::F
    value::V
    function InitialTFunction(f , t ,  ::Type{D} = Float64) where {D <:Number}
        fun = t -> f(t)
        F = typeof(fun)
        val = @. fun(t)
        V = typeof(val)
        new{D, F, V}(fun,val)
    end 
end
((init::InitialTFunction{D})(m::Int)::D) where D  = init.value[m]

struct BoundaryFunction{D, F, V, BCtype, BCdirection} <: AbstractHTFunction{D, F, V}
    fun::F
    value::V
    function BoundaryFunction(f , t, ::B , ::DIR,   ::Type{D} = Float64) where {D, B  <: BCtype, DIR <: AbstractBCDirection} where BCtype <: Union{DirichletBC,NeumanBC}
        fun = t -> f(t)
        val = @. fun(t)
        V = typeof(val)
        F = typeof(fun)
        new{D, F, V, B, DIR}(fun,val)
    end 
    function BoundaryFunction(f, t, ::B , ::DIR, ::Type{D} = Float64) where { D, B <: BCtype, DIR <: AbstractBCDirection} where BCtype <: RobinBC
        val = nothing
        V = typeof(val)
        fun = t -> f(t)
        F = typeof(fun)
        new{D, F, V, B, DIR}(fun,val)
    end 
end


"""
    (bc::BoundaryFunction{F,BCtype} )(m::Int) where {F,BCtype <: Union{DirichletBC,NeumanBC}}

For DirichletBC and NeumanBC the value of BC can be evaluated in advance and collected by index
"""
(bc::BoundaryFunction{F,V,BCtype} )(m::Int) where {F,V,BCtype <: Union{DirichletBC,NeumanBC}} = bc.value[m]