function [T,x,t,maxFn] = CN_case3_dirichle(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
% BDF1 = CN + explicit
% C_f - thermal capacity function (cp*Ro)
% L_f - thermal conductivity function
% Ld_f - thermal conductivity derivative 
% H - thickness
% tmax - time interval
% initT_f - initial temperature distribution function 
% BC_up_f - upper BC function of time
% BC_dwn_f - dwn BC function of time
% N,M - spatial and time discretization
x = linspace(0,H,N)';dx = x(2) - x(1);
t = linspace(0,tmax,M)';dt = t(2) - t(1);

T = zeros(N,M); % columns - distribution, rows time
T(:,1) = initT_f(x); % applying initial conditions
T(1,:) = BC_up_f(t); % applying upper BC
T(N,:) = BC_dwn_f(t); % applying lower BC
dd = dt/(dx*dx);
maxFn = 0;
C = zeros(N); %LHS matrix
Nm = zeros(N); %RHS matrix
D = DiffMat2(N,N); % finite difference matrix 

im1 = diagonal_inds(-1,N, N); % индексы -1-й диагонали
ip1 = diagonal_inds(1,N, N); % индексы +1-й диагонали
ip0 = diagonal_inds(0,N, N); % индексы 0-й диагонали
    for m = 1:M-1
        % filling material properties vectors
        lam_m = L_f(T(:,m));% thermal conductivity evaluated for the m'th timestep
        Cm = C_f(T(:,m));
        am = lam_m./Cm;
        Fm = dd*am;% Fm
        phi_m = Ld_f(T(:,m))./(lam_m*4); % phi
        b = Fm.*phi_m.*(D*T(:,m)) .^2;

        % filling LHS matrix
        C(ip0) = 1 + Fm;
        C(ip1) = - Fm(1:end-1)/2;
        C(im1) = - Fm(2:end)/2;
        % filling RHS matrix
        Nm(ip0) = 1 - Fm;
        Nm(ip1) = Fm(1:end-1)/2;
        Nm(im1) = Fm(2:end)/2;            

        % applying BC to make first row equation trivial
        b(1) = T(1,m + 1) - Nm(1,:)*T(:,m);
        b(end) = T(end,m + 1) - Nm(end,:)*T(:,m);
        C(1) = 1;C(1,2) = 0;C(1,3) = 0;
        C(end) = 1;C(end,end - 1) = 0;C(end,end - 1) = 0;
        % solving
        T(:,m + 1) = C\(Nm*T(:,m) + b);
    end
end

