module num_grad_check
using LinearAlgebra
function num_grad(f,x)
	fi = f(x)
	g = Vector{eltype(x)}(undef,length(x))
	h = 50*eps()
	for i in eachindex(x)
		f_dim!(g,f,x,i,h)
	end
	return @. g=(g - fi)/h 
end
function f_dim!(g,f,x,i,h)
	x[i]+=h
	g[i] = f(x)
	x[i]-=h;
	return nothing
end
function num_grad_copy(f,x)
	fi = f(x)
	g = Vector{eltype(x)}(undef,length(x))
	h = 50*eps()
	for i in eachindex(x)
        p = copy(x)
        p[i] += h
		g[i] = (f(p) - fi)/h
	end
	return g
end
function matrix_noarg(f,x)
    h = 50*eps()
	return map(f,eachcol(x*ones(length(x))' + h*I(length(x))))/h .- f(x)/h
end
end
using BenchmarkTools
f(x) = *(x...)
@benchmark num_grad_check.num_grad(f,rand(3))
@benchmark num_grad_check.matrix_noarg(f,rand(3))
@benchmark num_grad_check.num_grad_copy(f,rand(3))