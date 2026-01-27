# PolynomialWrappers
using LinearAlgebra,Interpolations,Polynomials,LegendrePolynomials,StaticArrays,RecipesBase

const POLY_NAMES_TYPES_DICT = Base.ImmutableDict(
    :trig => :TrigPolyWrapper, 
    :leg => :LegPolyWrapper,
    :stand => :StandPolyWrapper ,
    :chebT => :ChebPolyWrapper,
    :bernstein => :BernsteinPolyWrapper,
    :bernsteinsym => :BernsteinSymPolyWrapper
)
abstract type AbstractPolyWrapper{P,V,T} end
for (_poly_name, _PolyType) in  POLY_NAMES_TYPES_DICT
    x = String(_poly_name)
    @eval struct $_PolyType{P,T} <: AbstractPolyWrapper{P,T,Symbol($x)}
        coeffs::MVector{P,T}
    end
end

const SUPPORTED_POLYNOMIAL_TYPES = Base.ImmutableDict([k=>eval(d) for (k,d) in  POLY_NAMES_TYPES_DICT]...)
                #=:stand=> StandPolyWrapper,#Standard basis polynomials from Polynomials.Polynomial,
                :chebT=> ChebPolyWrapper,# Chebyshev polynomials from Polynomials.ChebyshevT,
                :trig => TrigPolyWrapper, # Self-made primitive type for trigonometric functions
                :leg =>  LegPolyWrapper, # Legendre polynomials         
                :bernstein => BernsteinPolyWrapper  =#
#)
#=struct TrigPolyWrapper{P,T} <: AbstractPolyWrapper{P,T,:trig}
    coeffs::MVector{P,T}
end
struct LegPolyWrapper{P,T}  <: AbstractPolyWrapper{P,T,:leg}
    coeffs::MVector{P,T}
end
struct StandPolyWrapper{P,T}  <: AbstractPolyWrapper{P,T,:stand} 
    coeffs::MVector{P,T}
end
struct ChebPolyWrapper{P,T}  <: AbstractPolyWrapper{P,T,:chebT}
    coeffs::MVector{P,T}
end
struct BernsteinPolyWrapper{P,T} <: AbstractPolyWrapper{P,T,:bernstein}
    coeffs::MVector{P,T}
end=#
(::Type{P})(x::Vector{T}) where {P<:AbstractPolyWrapper,T} = P{length(x),T}(MVector{length(x)}(x))
(::Type{P})(x::NTuple{N,T}) where {P<:AbstractPolyWrapper,T,N} = P{N,T}(MVector(x))
poly_name(::P) where P<: AbstractPolyWrapper{N,T,V} where {N,T,V} = V
poly_name(::Type{P}) where P<: AbstractPolyWrapper{N,T,V} where {N,T,V} = V
poly_degree(::AbstractPolyWrapper{N}) where {N} = N - 1
poly_degree(::Type{P}) where P<:AbstractPolyWrapper{N} where {N} = N - 1
parnumber(::AbstractPolyWrapper{N,T,V}) where {N,T,V} = N


"""
    Function to evaluate polynomials
"""
eval_poly(poly::StandPolyWrapper,x) = Polynomials.Polynomial(poly.coeffs)(x)
eval_poly(poly::ChebPolyWrapper,x) = Polynomials.ChebyshevT(poly.coeffs)(x)
eval_poly(::LegPolyWrapper,degree,x) = LegendrePolynomials.Pl(x,degree)
function eval_poly(::TrigPolyWrapper,degree,x) 
     degree != 0 || return 1
     n = 1 + floor(degree/2) 
     return isodd(degree) ? sin(n*pi*x) : cos(n*pi*x)
end
function (poly::Union{StandPolyWrapper,ChebPolyWrapper})(x::Number)
    return eval_poly(poly,x)
end
"""

"""
function eval_poly(::BernsteinPolyWrapper{D},k::Int,x::Number) where D # D - number of polynomial coefficients
    d = D - 1 # polynomial degree
    return binomial(d,k)* ^(1.0 - x, d - k) * x^k
end
function eval_poly(::BernsteinSymPolyWrapper{D,T},k::Int,x::Number,a::T=-1.0,b::T=1.0) where {D,T} # D - number of polynomial coefficients
    d = D - 1 # polynomial degree
    s = b - a
    return binomial(d,k)* ^((b - x)/s, d - k) * ^((x - a)/s,k)
end
"""
    bern_max(::BernsteinPolyWrapper{D},k::Int)

Returns Bernstein's monomial (maximum_value, maximum_location) tuple
"""
function bern_max(::Type{BernsteinPolyWrapper{D,T}},k::Int) where {D,T}
    d = D - 1
    return (^(k,k)* ^(d, -T(d))*^(d-k,d-k)*binomial(d,k), k/d)
end
function bern_max(::Type{BernsteinSymPolyWrapper{D,T}},k::Int,a::T=-1.0,b::T=1.0) where {D,T}
    d = D - 1
    s = b - a
    return (^(k,k)* ^(d,-T(d))*^(d-k,d-k)*binomial(d,k), s*k/d + a)
end
"""
(poly::Union{LegPolyWrapper,TrigPolyWrapper})(x::Number)

    Function returns the summation of polynomials values with coefficients 
    LegPolyWrapper wraps the LegendrePolynomials.jl library function to 
    make it consistent with Polynomials.jl 
    TrigPolyWrapper simple type for trigonometric function polynomils
"""
function (poly::Union{LegPolyWrapper{N},TrigPolyWrapper{N},BernsteinPolyWrapper{N},BernsteinSymPolyWrapper{N}})(x::Number) where N
    #LegendrePolynomials.Pl(x,l) - computes Legendre polynomial of degree l at point x 
    #=res = 0.0
    for i ∈ 1:N
        coeff = poly.coeffs[i]
        coeff != 0.0 || continue
        res += coeff*eval_poly(poly, i - 1,x)
    end
    return res =#
    return sum(ntuple(i -> poly.coeffs[i]*eval_poly(poly,i - 1,x),N))
end
"""
    VanderMatrix{M <: SMatrix , R <: SMatrix, V <: SVector}
    
This type stores the Vandemonde matrix (fundamental matrix of basis functions),
supports various types of internal polynomials 
Structure VanderMatrix has the following fields:
    v - the matrix itself (each column of this matrix is the value of basis function)
    v_unnorm - version of matrix with unnormalized basis vectors (used for annormalized coefficients of polynomial fitting)
    x_first -  first element of the initial vector 
    x_last  -  the last value of the initial vector
    xi - normalized vector 
    poly_type  - polynomial type name (nothing depends on this name)
"""
struct VanderMatrix{N,CN,T,NxCN,CNxCN,P} #M <: SMatrix , R <: SMatrix, V <: SVector}
    v::SMatrix{N,CN,T,NxCN} # matrix of approximating functions 
    v_unnorm::SMatrix{N,CN,T,NxCN} # unnormalized vandermatrix used to convert fitted parameters if necessarys
    # QR factorization matrices
    Q::SMatrix{N,CN,T,NxCN} 
    R::SMatrix{CN,CN,T,CNxCN}
    x_first::T # first element of the initial array
    x_last::T # normalizing coefficient 
    xi::SVector{N,T} # normalized vector-column 
end# struct spec
"""
    bern_max(::VanderMatrix{N,CN,T,NxCN,CNxCN,P}) where {N,CN,T,NxCN,CNxCN,P<:BernsteinPolyWrapper{CN}}

Returns a vector of maximal values of Bernstein basis polynomial basis for particular VanderMatrix
"""
bern_max_values(::VanderMatrix{N,CN,T,NxCN,CNxCN,P}) where {N,CN,T,NxCN,CNxCN,P<:Union{BernsteinPolyWrapper{CN},BernsteinSymPolyWrapper{CN}}} = [bern_max(P,i)[1] for i in 0:CN-1]
bern_max_locations(::VanderMatrix{N,CN,T,NxCN,CNxCN,P}) where {N,CN,T,NxCN,CNxCN,P<:Union{BernsteinPolyWrapper{CN},BernsteinSymPolyWrapper{CN}}} = [bern_max(P,i)[2] for i in 0:CN-1]
"""
    VanderMatrix(x::StaticArray{Tuple{N},T,1},
                      PolyWrapper::Type{P} # = StandPolyWrapper{CN}
                    ) where {N, T, P <:AbstractPolyWrapper{CN,PN}} where {CN,PN}

Input: 
    x  - vector of independent variables (coordinates)
    Val(CN) - vandermatrix column size (degree of polynomial + 1)
    poly_type - polynomial type name, must be member of SUPPORTED_POLYNOMIAL_TYPES

"""
function VanderMatrix(x::StaticArray{Tuple{N},T,1},
                      PolyWrapper::Type{P} # = StandPolyWrapper{CN}
                    ) where {N, T, P <:AbstractPolyWrapper{CN,PN}} where {CN,PN}
            (_xi,x_first,x_last) = normalize_x(x)
            V = Matrix{T}(undef,N,CN) 
            Vunnorm = Matrix{T}(undef,N,CN)  # T{length(x)}(MVector{length(x)}(x))
            poly_obj = PolyWrapper(MVector{CN}(zeros(CN)))
            _fill_vander!(V, poly_obj,_xi)
            _fill_vander!(Vunnorm, poly_obj,x)
            NxCN =  N*CN     
            MatrixType  = SMatrix{N, CN, T,NxCN}
            CNxCN =  CN * CN
            RMatrixType = SMatrix{CN, CN, T,CNxCN}
            VectorType = SVector{N,T}
            _V = MatrixType(V)
            (Q,R) = qr(_V)
            VanderMatrix{N,CN,T,NxCN,CNxCN,PolyWrapper}(_V,# Vandermonde matrix
                MatrixType(Vunnorm), #unnormalized vandermatrix
                MatrixType(Q),
                RMatrixType(R),
                x_first, # first element of the initial array
                x_last, # normalizing coefficient 
                VectorType(_xi),  
            )
end
poly_name(::VanderMatrix{N,CN,T,NxCN,CNxCN,P}) where {N,CN,T,NxCN,CNxCN,P} = poly_name(P)
"""
    _fill_vander!(V, poly_obj::AbstractPolyWrapper,xi)

Function to fill the matrix V columns from polynomial basis functions constructor
with argument vector xi 
"""
function _fill_vander!(V, poly_obj::AbstractPolyWrapper,xi)
    VW = @views eachcol(V)
    for (i,col) ∈ enumerate(VW)
        poly_obj.coeffs[i] = 1.0                              
        @. col = poly_obj(xi)
        poly_obj.coeffs[i] = 0.0
    end  
    return poly_obj
end
"""
    is_the_same_x(v::VanderMatrix,x::AbstractVector)

Checks if input `x` is the same as the one used for VanderMatrix creation
"""
function is_the_same_x(vander::VanderMatrix{N,CN,T},x::AbstractVector{T}) where {N,CN,T}
    (length(x) == N && issorted(x) ) || return false
    x_f = first(x)
    vander.x_first == x_f || return false
    x_l = last(x)
    vander.x_last == x_l || return false
    for i in 1:N 
        x[i] == denormalize_x(vander.xi[i], x_f, x_l)  || return false
    end
    return true
end
"""
    *(V::VanderMatrix,a::AbstractVector)

VanderMatrix object can be directly multiplyed by a vector
"""
Base.:*(V::VanderMatrix,a::AbstractVector) =V.v*a 
"""
    polyfit(V::VanderMatrix{N,CN,T},x::VT,y::VT) where {N,CN,T<:Number,VT<:Vector{T}}

Fits data x - coordinates, y - values using the VanderMatrix
basis function (this coefficients for normalized x-vector)
```julia
    (a,y_fitted, gf) = polyfit(V,x,y)
    y_fitted =  V.v*a # V.v - is the vandermonde matrix
    # for normalized x, which has all values within [-1,1] range
    # if a = [a₁ , ..., aₙ], 
    # e.g if V is for standard basis:
    (xnorm,) = normalize_x(x) # returns vector 
    # y_fitted = a₁ + a₂xnorm + ... + aₙxnormⁿ⁻¹
```

Input:
    x - coordinates, [Nx0]
    y - values, [Nx0]
returns tuple with vector of polynomial coefficients, values of y_fitted at x points
and the norm of goodness of fit     
"""
function polyfit(V::VanderMatrix{N,CN,T},x::VT,y::VT) where {N,CN,T<:Number,VT<:Vector{T}}
    yi =  !is_the_same_x(V,x) ? linear_interpolation(x,y)(denormalize_x(V)) : y
    a =SVector{CN,T}(V.R\(transpose(V.Q)*yi)) # calculating pseudo-inverse
    y_fit = V*a
    goodness_fit = norm(yi .- y_fit)
    return  (a, y_fit, goodness_fit) 
end
"""
    polyfitn(V::VanderMatrix{N,CN,T},x::VT,y::VT) where {N,CN,T<:Number,VT<:Vector{T}}

Fits data x - coordinates, y - values using the VanderMatrix
basis function (coefficients for unnormalized x-vector)

Input:
    x - coordinates, [Nx0]
    y - values, [Nx0]
returns tuple with vector of polynomial coefficients, values of y_fitted at x points
and the norm of goodness of fit  
"""
function polyfitn(V::VanderMatrix{N,CN,T},x::VT,y::VT) where {N,CN,T<:Number,VT<:Vector{T}}
    yi =  !is_the_same_x(V,x) ? linear_interpolation(x,y)(denormalize_x(V)) : y
    (Q,R) = qr(V.v_unnorm)
    a =SVector{CN,T}(R\transpose(Q)*yi)# calculating pseudo-inverse
    y_fit = V.v_unnorm * a
    goodness_fit = norm(yi .- y_fit)
    return  (a, y_fit, goodness_fit) 
end

"""
    normalize_x(x::AbstractVector)

Makes all elements of vector x to fit in range -1...1
returns normalized vector , xmin and xmax values
All elements of x must be unique
Makes all elements of vector x to fit in range -1...1
returns normalized vector , xmin and xmax values
All elements of x must be unique

"""
function normalize_x(x::AbstractVector)
    @assert allunique(x) "All elements of input vector must be unique"
    _x = !issorted(x) ? sort(x) : x

    x_min = _x[1]
    x_max = _x[end]
    a = 2.0/(x_max - x_min)
    xi_converted  = @. a*(_x - x_min) - 1.0
    return (xi_converted ,x_min,x_max)
end

"""
    denormalize_x(normalized_x::AbstractVector, x_min,x_max)

Creates normal vector from one created with [`normalize_x`](@ref)` function 
"""
function denormalize_x(normalized_x::Number, x_min, x_max)
    return 0.5*(normalized_x + 1.0)*(x_max - x_min) + x_min
end
denormalize_x(V::VanderMatrix) = Vector(denormalize_x.(V.xi, V.x_first,V.x_last))
function triplicate_columns(a::AbstractVector,T)
    return T(repeat(a,1,3))
end

"""
    fill_box_constraint!(lb,ub,::VanderMatrix{N, CN, T, NxCN, CNxCN, P},
                val_bounds::NTuple{2,T}) where {N, CN, T, NxCN, CNxCN, P<:BernsteinSymPolyWrapper}

Evaluates box-boundaries for polynomial coefficients for `BernsteinSymPolyWrapper` 
polynomial basis
"""
function fill_box_constraint!(lb,ub,::VanderMatrix{N, CN, T, NxCN, CNxCN, P},
                val_bounds::NTuple{2,T}) where {N, CN, T, NxCN, CNxCN, P<:BernsteinSymPolyWrapper}
    fill!(lb,first(val_bounds))
    fill!(ub,last(val_bounds))
end
@recipe function f(m::AbstractPolyWrapper)
    minorgrid--> true
    gridlinewidth-->2
    dpi-->600
    return t->m.(t)
end


#p = plot(title = "Monomials",legend=:top,legend_columns=2, background_color_legend=RGBA(1, 1, 1, 0.0),foreground_color_legend=nothing)
@recipe function f(V::VanderMatrix{N,CN,T,NxCN,CNxCN,P}; infill = true) where {N,CN,T,NxCN,CNxCN,P}
    for (i,c) in enumerate(eachcol(V.v))
        @series begin 
            label:="$(i)"    
            linewidth:=2
            legend := :top
            legend_columns :=2

            foreground_color_legend :=nothing
            if infill
                fillrange:=0
                fillalpha:=0.3
            end
            markershape:=:none
            (V.xi, c)
        end
    end
end