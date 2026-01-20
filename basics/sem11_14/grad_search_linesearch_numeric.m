function [x,Fval,ii,flag]=grad_search_linesearch_numeric(x0,F,gradF,options)
% простой оптимизатор методом градиентного поиска
% входные аргументы:
%                   x0 - стартовая точка
%                   F - указатель на скалярную функцию векторного аргумента
%                   gradF - указатель на функцию расчета градиента функции
%                           F
%                   Опциональные аргументы в формате имя-значение
%                   mu (optional)- амплитудный коэффициент, длина шага 
%                   N (optional)- ограничение на число итераций
%                   tol (optional)- точность (относительное изменение для
%                   двух последовательных итераций)
    arguments
        x0 double
        F function_handle
        gradF function_handle
        options.mu (1,1) double =1e-2
        options.N (1,1) double =10000
        options.tol (1,1)double =1e-6
    end
    x=x0(:);mu = options.mu;N = options.N;tol = options.tol;flag=[true true];
    Fval=F(x0);ii=1;
    while ii<N && all(flag) % условием остановки служит достидение заданного числа итераций и проверка сходимости
        x_previous=x;
        F_previous = Fval; % рассчитываем значение функции
        grad_value = gradF(x); % рассчитываем градиент функции
        grad_norm = norm(grad_value);
        if grad_norm==0
            return
        end
        grad_direction = grad_value/grad_norm; % используем только направление градиента
        grad_direction = grad_direction(:);
        F_mu = @(mu_trial) F(x - mu_trial*grad_direction);% формулируем как указатель на функцию от длины шага
        gradF_mu = @(mu_trial) num_grad(F_mu,mu_trial);
        [mu,~,iter_number]=grad_search_linesearch(mu,F_mu,gradF_mu,"N",20);
        %[mu,iter_number] = fminbnd(gradF_mu,1e-3,100*mu,optimset("MaxIter",20));
        Fval = F_mu(mu);
        x = x - mu*grad_direction;
        ii=ii+iter_number;
        flag = [norm(Fval-F_previous)>tol ...
            norm(x_previous-x)>tol...
            grad_norm>tol]; 
        
    end
end
