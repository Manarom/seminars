function [x,Fval,ii,flag,search_history]=grad_lsqr_linesearch(x0,r,Jac,options)
% простой оптимизатор методом градиентного поиска
% входные аргументы:
%                   x0 - стартовая точка алгоритма оптимизации [P,1] (P -
%                   число переменных оптимизации)
%                   r - указатель на векорную функцию, которая рассчитывает
%                   вектор r - вектор ошибки, возвращает вектор размером
%                   [N,1] - N - число экспериментальных точек
%                   Jac - указатель на функцию, которая рассчитывает
%                   матрицу Якоби, она возвращает матрицу [P,N]
%                   mu (optional)- амплитудный коэффициент, длина шага
%                   (learning rate)
%                   N (optional)- ограничение на число итераций
%                   tol (optional)- точность (относительное изменение для
%                   двух последовательных итераций)
% выходные аргументы:
%                   x - оптимальное значение вектора параметров оптимизации
%                   (минимизатор)
%                   Fval - значение функции для найденного минимизатора
%                   ii - число вычислений функции и ее градиента
%                   flag - флажок критериев сходимости (изменение значения функции, изменение аргумента функции, модуль градиента)
%                   search_history - матрица, у которой столбцы -
%                   координаты в пространстве оптимизации, по которым ходил
%                   алгоритм
%                   
    arguments
        x0 double
        r function_handle
        Jac function_handle
        options.mu (1,1) double =1e-2
        options.N (1,1) double =1000
        options.tol (1,1)double =1e-6
    end
    ii=1;
    x=x0(:);mu = options.mu;N = options.N;tol = options.tol;
    flag=[true true true]; % флажок показателей сходимости

    residual_vector=r(x0); % вектор ошибки
    J = Jac(x0); % матрица Якоби
    Fval = F(residual_vector); % значение функции невязки

    is_return_search_history = false; % флажок, который показывает нужно ли нам возвращать историю поиска
    if nargout==5 % так как хранение всех точек может быть тяжелым делаем 
    % матрицу только если функция вызвана с пятью выходными аргументами
        is_return_search_history =true;% если число выходных аргументов равно пяти, то значит нужно сохранить историю
        search_history = NaN(numel(x),N);% резервируем память под все точки алгоритма
        search_history(:,1) = x0;% записываем стартовую точку в историю поиска
    end
    % основной цикл поиска 
    while ii<N && all(flag) % условием остановки служит достижение заданного числа итераций и проверка сходимости
        x_previous=x;
        F_previous = Fval; % значения коордианты и функции на предыдущей итерации
        grad_value = J*residual_vector; % рассчитываем градиент функции

        grad_norm = norm(grad_value); % модуль градиента 
        if grad_norm==0 % если градиент равен нулю (внезапно), то мы уже в точке экстремума
            return
        end
        grad_direction = grad_value/norm(grad_value); % используем только направление градиента
        
        
        F_mu = @(mu_trial) r(x - mu_trial*grad_direction)'*r(x - mu_trial*grad_direction)/2;% формулируем как указатель на функцию от длины шага
        
        
        [mu,~,~]=fminbnd(F_mu,0,10,optimset("MaxIter",20,"Display","off"));

        x= x - mu*grad_direction(:);% рассчитываем координату для следующей точки

        residual_vector = r(x); % обновляем вектор ошибки
        J = Jac(x); % обновляем матрицу Якоби
        Fval=F(residual_vector); % рассчитываем значение функции для этой координаты
        
        if is_return_search_history
            search_history(:,ii+1) = x;% если нужны промежуточные точки - добавляем
        end
        % флажок проверки сходимости
        flag = [norm(Fval-F_previous)>tol ...%  изменение значения функции
            norm(x_previous-x)>tol ...%  изменение координаты
            grad_norm>tol]; %  модуль градиента
        ii=ii+1; 
    end
    if is_return_search_history
        search_history = search_history(:,1:ii);
    end
end
function discrepancy = F(r)
    discrepancy = 0.5*norm(r)^2;
end