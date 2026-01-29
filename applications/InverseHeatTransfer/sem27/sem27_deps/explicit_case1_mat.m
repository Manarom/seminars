function [T,x,t,maxFn] = explicit_case1_dirichle2(C_f, L_f,Ld_f, H, tmax,initT_f,BC_up_f,BC_dwn_f,M,N)
% solves heat transfer equation using BFD1 - explicit - explicit scheme
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

T = zeros(N,M); % колонка - распределение температуры по тощине в заданный момент времени
T(:,1) = initT_f(x);
T(1,:) = BC_up_f(t); % applying upper BC
T(N,:) = BC_dwn_f(t); % applying lower BC
dd = dt/(dx*dx);
maxFn = 0;
A = zeros(N);
im1 = diagonal_inds(-1,N, N); % индексы -1-й диагонали
ip1 = diagonal_inds(1,N, N); % индексы +1-й диагонали
ip0 = diagonal_inds(0,N, N); % индексы 0-й диагонали
D = DiffMat2(N,N);
    for m = 1:M-1
        lam_m = L_f(T(:,m));% thermal conductivity evaluated for the m'th timestep
        Cm = C_f(T(:,m));
        am = lam_m./Cm;
        Fm = dd*am;% Fm
        phi_m = Ld_f(T(:,m))./(lam_m*4); % phi
        % граничные условия
        A(ip0) = 1 - 2*Fm;
        A(ip1) = Fm(1:end-1);
        A(im1) = Fm(2:end);
        % нелинейная часть 
        b = Fm.*phi_m.*(D*T(:,m)) .^2;
        % модифицируем вектор b, чтобы не менять размер матрицы 
        b(1) = T(1,m+1)  - A(1,:)*T(:,m);
        b(end) = T(end,m+1)  - A(end,:)*T(:,m);
        % решение
        T(:,m+1) = A*T(:,m) + b;

    end
end
