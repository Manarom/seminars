abstract type AbstractGrid{N,M,T}  end
timestep(::AbstractGrid,::Int) = ni_err()
xstep(::AbstractGrid,::Int) =  ni_err()
tvalue(::AbstractGrid,::Int) =  ni_err()
xvalue(::AbstractGrid,::Int) =  ni_err()
datatype(::AbstractGrid{N,M,T}) where {N,M,T} = T

firstx(::AbstractGrid) = 0.0
firstt(::AbstractGrid) = 0.0 
lastx(g::AbstractGrid{N}) where {N} = xvalue(g,N)
lastt(g::AbstractGrid{N,M}) where {N,M} = tvalue(g,M)

xrange(g::AbstractGrid{N}) where N= range(firstx(g),lastx(g),length = N)
trange(g::AbstractGrid{N,M}) where {N,M} = range(firstt(g),lastt(g),length = M)
tpoints(::AbstractGrid{N,M}) where {N,M} = M
xpoints(::AbstractGrid{N}) where {N}= N

Base.size(::AbstractGrid{N,M}) where {N,M} = (N,M)
function Base.size(::AbstractGrid{N,M}, d::Int) where {N,M}
    d < 1 && error("arraysize: dimension out of range")
    return d == 1 ? N : M 
end   
abstract type AbstractGridIterator{T} end
# iterator over time 
struct EachTimeStep{T} <: AbstractGridIterator{T}
    g::T 
    EachTimeStep(g::T) where T <: AbstractGrid = new{T}(g)
end
eachtime(g::AbstractGrid) = EachTimeStep(g)
Base.length(g::EachTimeStep) = tpoints(getfield(g,:g))
Base.iterate(itr::EachTimeStep) = (firstt(itr.g) , 2)
Base.iterate(itr::EachTimeStep,state::Int) = 1 <= state <= tpoints(itr.g) ? (tvalue(itr.g, state), state + 1) : nothing

struct EachXStep{T} <: AbstractGridIterator{T}
    g::T 
    EachXStep(g::T) where T <: AbstractGrid = new{T}(g)
end
eachx(g::AbstractGrid) = EachXStep(g)
Base.length(g::EachXStep) = xpoints(getfield(g,:g))

Base.iterate(itr::EachXStep) = (firstx(itr.g) , 2)
Base.iterate(itr::EachXStep, state::Int) = (1 <= state && state <= xpoints(itr.g)) ? (xvalue(itr.g, state), state + 1) : nothing

function Base.map!(f, dest::AbstractVector, itr::AbstractGridIterator )
    @assert length(dest) == length(itr)    
    for (i,x) in enumerate(itr)
        @inbounds dest[i] = f(x)
    end
    return dest
end
struct UniformGrid{N,M,T} <: AbstractGrid{N,M,T}
    dx::T
    dt::T
    UniformGrid(xmax::T, tmax::T,::Val{N},::Val{M}) where {T,N,M} = new{N,M,T}(xmax/( N  - 1) , tmax/(M - 1) ) 
end

timestep(g::UniformGrid, ::Int) = g.dt
xstep(g::UniformGrid, ::Int) = g.dx 
tvalue(g::UniformGrid, m::Int) = g.dt * (m - 1)
xvalue(g::UniformGrid, n::Int) = g.dx * (n - 1)