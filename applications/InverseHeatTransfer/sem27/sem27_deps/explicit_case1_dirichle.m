function [T,x,t,maxFn] = explicit_case1_dirichle(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
% C_f - thermal capacity function (cp*Ro)
% L_f - thermal conductivity function
% Ld_f - thermal conductivity derivative 
% H - thickness
% tmax - time interval
% initT_f - initial temperature distribution function 
% BC_up_f - upper BC function of time
% BC_dwn_f - dwn BC function of time
% N,M - spatial and time discretization
x = linspace(0,H,N)';
t = linspace(0,tmax,M)';
dx = x(2) - x(1);
dt = t(2) - t(1);
T = zeros(N,M); % columns - distribution, rows time
T(:,1) = initT_f(x);
T(1,:) = BC_up_f(t); % applying upper BC
T(N,:) = BC_dwn_f(t); % applying lower BC
dd = dt/(dx*dx);
maxFn = 0;
    for m = 1:M-1
        lam_m = L_f(T(:,m));% thermal conductivity evaluated for the m'th timestep
        Cm = C_f(T(:,m));
        am = lam_m./Cm;
        Fm = dd*am;% Fm
        phi_m = Ld_f(T(:,m))./(lam_m*4); % phi
        for n = 2:N-1
            r = Fm(n);
            r1 = 1 - 2*r;
            fi = phi_m(n);
            b_n = r*fi*(T(n - 1,m) - T(n + 1,m))^2 ; % [\vec{b}^m(\vec{T}^m)]_n= F_{n}^m \phi_{n}^m (T^{m}_{n-1} - T^{m}_{n+1})^2
            T(n,m + 1) = r*(T(n - 1,m) + T(n + 1,m)) + r1*T(n,m) + b_n;
            if r > maxFn
                maxFn = r;
            end
        end
    end
end

