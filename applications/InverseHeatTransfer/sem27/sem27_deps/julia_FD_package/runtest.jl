N = 400000

f = rand(N)
fm1 = @view f[2 : end]
fp1 = @view f[1 : end - 1]
include("finite_difference_functions.jl")
using LinearAlgebra,BenchmarkTools
using Test
import .OneDHeatTransfer as OD
M = Tridiagonal(collect(fm1), collect(f), collect(fp1))

b = rand(N)

c = copy(b)
b1 = copy(b)
b2 = copy(b)

mul!(c,M,b)
N > 5 || @show c


a0 = 0.0
am1 = 1.0
ap1 = 1.0
a = 1.0


@testset begin
    OneDHeatTransfer.tridiag_mul!(b1,fm1,f, fp1, a0, am1, a, ap1,N)
    N > 5 ||@show b1
    OneDHeatTransfer.column_sym_tridiag_mul!(b2, f, a0, a , ap1, N)
    N > 5 ||@show b2
    @test norm(b1 - c) ≈ 0 atol=1e-10
    @test norm(b2 - c) ≈ 0 atol=1e-10
end

print("mul!")
@btime mul!(c,M,b)
print("OneDHeatTransfer.tridiag_mul!")
@btime OneDHeatTransfer.tridiag_mul!(b1,fm1,f, fp1, a0, am1, a, ap1,N)
print("OneDHeatTransfer.column_sym_tridiag_mul!")
@btime OneDHeatTransfer.column_sym_tridiag_mul!(b,f,a0,a,ap1,N) 

#=
@profview for _ in 1:1000
    test_mult!(b2,b,LHS,fm1,f, fp1, a0, am1, a, ap1,N)
end
=#

#=@profview for _ in 1:100000
    OneDHeatTransfer.tridiag_mul!(b1,fm1,f, fp1, a0, am1, a, ap1,N)
end=#
#=
@profview for _ in 1:100000
    mul!(c,M,b)
end

=#
#@code_warntype OneDHeatTransfer.column_sym_tridiag_mul!(b,f,a0,a,ap1,N) 

y = rand(N)
b = rand(N)
q0 = copy(b)
q1 = copy(b)
q2 = copy(b)

mul!(q0,M,b,1.0,1.0)
N > 5 || @show q0


a0 = 0.0
am1 = 1.0
ap1 = 1.0
a = 1.0


@testset begin
    OneDHeatTransfer.tridiag_muladd!(q1,b,fm1,f, fp1, a0, am1, a, ap1,N)
    N > 5 ||@show q1


    OneDHeatTransfer.column_sym_tridiag_muladd!(q2,b, f, a0, a , ap1, N)
    N > 5 ||@show q2
 
    @test norm(q1 - q0) ≈ 0 atol=1e-10
    @test norm(q2 - q0) ≈ 0 atol=1e-10    
end

print("mul!")
@btime mul!(q0,M,b,1.0,1.0)
print("OneDHeatTransfer.tridiag_muladd!")
@btime OneDHeatTransfer.tridiag_muladd!(q1,b,fm1,f, fp1, a0, am1, a, ap1,N)
print("OneDHeatTransfer.column_sym_tridiag_muladd!")
@btime OneDHeatTransfer.column_sym_tridiag_muladd!(q2,b, f, a0, a , ap1, N)


@profview for _ in 1:1000 
    OneDHeatTransfer.column_sym_tridiag_muladd!(q2,b, f, a0, a , ap1, N)
end