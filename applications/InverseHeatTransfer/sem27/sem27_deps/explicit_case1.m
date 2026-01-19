function [T,x,t,maxFn] = explicit_case1(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N, BC_UP_TYPE, BC_DWN_TYPE)
% C_f - thermal capacity function (cp*Ro)
% L_f - thermal conductivity function
% Ld_f - thermal conductivity derivative 
% H - thickness
% tmax - time interval
% initT_f - initial temperature distribution function 
% BC_up_f - upper BC function of time
% BC_dwn_f - dwn BC function of time
% N,M - spatial and time discretization
% BC_UP_TYPE - upper boundary conditions 1,2 or 3
% BC_DWN_TYPE - lower BC 1, 2 or 3 (heat flux is provided as a function of
% temperature)
    x = linspace(0,H,N)';
    t = linspace(0,tmax,M)';
    dx = x(2) - x(1);
    dt = t(2) - t(1);
    T = zeros(N,M); % columns - distribution, rows time
    T(:,1) = initT_f(x);
    if BC_UP_TYPE==1
        T(1,:) = BC_up_f(t); % applying upper BC
    elseif BC_UP_TYPE==2
        q = BC_up_f(t);
    end
    if BC_DWN_TYPE==1
        T(N,:) = BC_dwn_f(t); % applying lower BC
    elseif BC_DWN_TYPE==2
        g = BC_dwn_f(t); % dwn HF
    end
    dd = dt/(dx*dx);
    maxFn = 0;
    for m = 1:M-1
        lam_m = L_f(T(:,m));% thermal conductivity evaluated for the m'th timestep
        Cm = C_f(T(:,m));
        am = lam_m./Cm;
        Fm = dd*am;% Fm
        phi_m = Ld_f(T(:,m))./(lam_m*4); % phi
        if (BC_UP_TYPE == 2) || (BC_UP_TYPE == 3) % Neuman or Robin BC on the upper boundary
            if BC_UP_TYPE == 2
                hf = q(m);
            else
                hf = BC_up_f(T(1,m));
            end
            T0 = T(2,m) - (2*dx*hf/lam_m(1)); % T^m_2 - 2\Delta x \frac{q^m}{\lambda_1^m}                       
            T(1,m + 1) = iteration_step(Fm(1),phi_m(1),T0,T(1,m),T(2,m));
        end
        for n = 2 : N-1 % goes from 2 to N-1 coordinate nod
            r = Fm(n);
            T(n,m + 1) = iteration_step(r,phi_m(n),T(n - 1,m),T(n,m),T(n + 1,m));
            if r > maxFn
                maxFn = r;
            end
        end
        if BC_DWN_TYPE == 2 || BC_DWN_TYPE == 3% Neuman BC on the lower boundary 
            if BC_DWN_TYPE == 2
                hf = g(m);
            else
                hf = BC_dwn_f(T(end,m));
            end
            TNp1 =  (2*dx*hf/lam_m(end)) + T(end - 1,m); %                       
            T(end,m + 1) = iteration_step(Fm(end),phi_m(end),T(end-1,m),T(end,m),TNp1);
        end
    end
end
function T = iteration_step(r,fi,T1,T2,T3)
        r1 = 1 - 2*r;
        b_n = r*fi*(T1 - T3)^2 ; %
        T = r*(T1 + T3) + r1*T2 + b_n;    
end

