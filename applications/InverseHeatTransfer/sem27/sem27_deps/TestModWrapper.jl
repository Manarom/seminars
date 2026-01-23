module TestModWrapper


function external(f)
    a = 0.0
    for i in 1:1000000
        a = internal(f,i)
    end
    return a
end
function external2(f)
    i = 1
    internal2() = internal(f,i)
    a = 0.0
    for i in 1:1000000
        a = internal2()
    end
    return a 
end
function external0(f)
    a = 0.0
    for i in 1:1000000
        a = f(i)
    end
    return a 
end
function internal(f,i)
    return f(i)
end

end

using BenchmarkTools,Polynomials
f_tes2= Polynomials.ImmutablePolynomial((1e-6))
@benchmark TestModWrapper.external0(f_tes)
 @benchmark TestModWrapper.external(f_tes)
@benchmark TestModWrapper.external2(f_tes)