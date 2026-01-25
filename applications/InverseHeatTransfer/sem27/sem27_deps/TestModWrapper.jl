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
    function external3(f)
        i = 1
        g(i) = i-> f(i)
        a = 0.0
        for i in 1:1000000
            a = g(i)
        end
        return a 
    end
    internal_v(i) = g(i)
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
    function external4(f)
        a = 0.0
        c = fw(f)
        for i in 1:1000000
            a = internal4(c,i)
        end
        return a
    end
    struct fw{F} f::F end
    (c::fw{F})(i) where F = c.f(i)
    internal4(c::fw,i) = c(i)
end

using BenchmarkTools,Polynomials
f_tes  = t -> t - 1e-6
@benchmark TestModWrapper.external0($f_tes)
@benchmark TestModWrapper.external($f_tes)
@benchmark TestModWrapper.external2($f_tes)
@benchmark TestModWrapper.external3($f_tes)
@benchmark TestModWrapper.external4($f_tes)

