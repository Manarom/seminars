classdef BernsteinBasis < AbstractPolynomial  

    methods
        function obj = BernsteinBasis(x,degree)
            obj@AbstractPolynomial(x,degree)
        end
        function val = monomial(~,n,i,x)
            [a,b] = AbstractPolynomial.scalers();
            val = internal_monomial(n,i,x,a,b);
        end
        function val = monomial_derivative(~,n,i,x)
            % evaluates Bernstein monomial derivative
            [a,b] = AbstractPolynomial.scalers();
            val = n*(internal_monomial(n - 1,i - 1,x,a,b) - internal_monomial(n - 1,i,x,a,b))/(b - a);
        end
        function coeffs_der = derivative_conversion(obj,coeffs)
            % n = numel(coeffs); % degree of initial polynomial is n - 1
            % coeffs_der = zeros(n - 1, 1);
            % degree = n - 1; % degree of the initial polynomial
            % for i = 0:(degree - 1) % index goes over the monomial degree, index in coeffs vector is i + 1
            %     coeffs_der(i + 1) = degree*(coeffs(i + 2) - coeffs(i + 1)); % b'(i) = degree*(b(i+1) - b(i)), for  i = 0...n-1
            % end
            n = length(coeffs) - 1;
            if n == 0
                coeffs_der = 0;
                return;
            end
            [a,b] = AbstractPolynomial.scalers();
            [xm, xx] = obj.x_scalers();
            if (a==xm) && (b==xx)
                coeffs_der = n *(coeffs(2:end) - coeffs(1:end-1))/(b - a);
            else
                coeffs_der = n *(coeffs(2:end) - coeffs(1:end-1))/(xx - xm); % this works well with my version of annormalized calculations
            end
        end
    end
end
function val = internal_monomial(n,i,x,a,b)
        if i > n || i < 0
            val = 0;
            return
        end
        %a = -1; b = 1; 
        s = b - a;
        val = nchoosek(n,i).* power((b - x)/s,n - i) .* power((x - a)/s,i);
end

