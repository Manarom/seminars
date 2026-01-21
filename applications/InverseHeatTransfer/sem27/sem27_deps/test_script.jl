N = 10000

f = rand(N)
fm1 = @view f[2 : end]
fp1 = @view f[1 : end - 1]
include("finite_difference_functions.jl")
using LinearAlgebra

M = Tridiagonal(collect(fm1), collect(f), collect(fp1))

b = rand(N)

c = copy(b)
b1 = copy(b)

mul!(c,M,b)

a0 = 0.0
am1 = 1.0
ap1 = 1.0
a = 1.0
OneDHeatTransfer.tridiag_mul!(b1,fm1,f, fp1, a0, am1, a, ap1,N)
b1



function test_mult!(b,c,LHS,fm1,f, fp1, a0, am1, a, ap1,N)
    OneDHeatTransfer.fill_tridiag!(LHS,fm1,f,fp1,a0,am1,a,ap1)
    mul!(b,LHS,c)
end

LHS = similar(M)
b2 = copy(b)
test_mult!(b2,b,LHS,fm1,f, fp1, a0, am1, a, ap1,N)

@show norm(b1 - c)
@show norm(b2 - c)

@btime mul!(c,M,b)
@btime test_mult!(b2,b,LHS,fm1,f, fp1, a0, am1, a, ap1,N)
@btime OneDHeatTransfer.tridiag_mul!(b1,fm1,f, fp1, a0, am1, a, ap1,N)


#=
@profview for _ in 1:1000
    test_mult!(b2,b,LHS,fm1,f, fp1, a0, am1, a, ap1,N)
end
=#

@profview for _ in 1:100000
    OneDHeatTransfer.tridiag_mul!(b1,fm1,f, fp1, a0, am1, a, ap1,N)
end
#=
@profview for _ in 1:100000
    mul!(c,M,b)
end

=#