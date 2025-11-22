classdef StandBasis < AbstractPolynomial
   
    methods
        function obj = StandBasis(x,degree)
            obj@AbstractPolynomial(x,degree)
        end
        function val = monomial(~,~,i,x)
            if i == 0
                val = ones(numel(x),1);
                return
            end
            if i < 0
                val = zeros(numel(x),1);
                return 
            end
            val = x.^i;
        end
        function val = monomial_derivative(obj,~,i,x)
            % standard bassis monomial derivative
            val = i*monomial(obj,1,i-1,x);
        end
        function coeffs_der = derivative_conversion(obj,coeffs)
            % coeffs - coefficients of the initial polynomial
            n = numel(coeffs);
            coeffs_der = zeros(n - 1, 1);
            [xm, xx] = obj.x_scalers();
            [a,b] = AbstractPolynomial.scalers();
            for i = 0:n-2
                coeffs_der(i + 1) = (b - a)*(i + 1)*coeffs(i + 2)/(xx -xm);
            end
        end
    end
end
