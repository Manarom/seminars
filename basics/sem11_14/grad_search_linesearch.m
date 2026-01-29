function [x,Fval,ii,flag,search_history]=grad_search_linesearch(x0,F,gradF,options)
% простой оптимизатор методом градиентного спуска с линейным поиском
% параметры см GRAD_SEARCH 
    arguments
        x0 double
        F function_handle
        gradF function_handle
        options.mu (1,1) double =1e-2
        options.N (1,1) double =10000
        options.tol (1,1)double =1e-6
        options.alfa (1,1) double {mustBeInRange(options.alfa,1e-4, 100)}=2 % коэффициент расширения
        options.beta (1,1) double {mustBeInRange(options.beta,1e-4, 100)}=0.5 % коэффициент сжатия
        options.tries (1,1) double {mustBeInteger, mustBePositive} = 10 % число пробных пристрелок
    end
    is_return_search_history = false;x=x0(:);mu = options.mu;N = options.N;tol = options.tol;flag=[true true];alfa = options.alfa;beta = options.beta;
    tries = options.tries;
    if nargout==5 % так как хранение всех точек может быть тяжелым
        is_return_search_history =true;% если число выходных аргументов равно пяти, то значит нужно сохранить историю
        search_history = NaN(numel(x),N+1);% резервируем память под все точки алгоритма
        search_history(:,1) = x0;
    end
    
    Fval=F(x0);ii=1;
    while ii<N && all(flag) % условием остановки служит достидение заданного числа итераций и проверка сходимости
        x_previous=x;
        F_previous = Fval; % рассчитываем значение функции
        grad_value = gradF(x); % рассчитываем градиент функции
        grad_norm = norm(grad_value); % модуль градиента
        if grad_norm==0
            return
        end
        grad_direction = grad_value/grad_norm; % используем только направление градиента
        grad_direction = grad_direction(:);
        jj=0;% счетчик триальных итераций
        Fval_trial=Fval;% стартовые 
        mu_trial = mu;
        while (jj<=tries)% в этом цикле производим варьирования длины шага вдоль градиента
                %x_previous_trial=x_trial;
                Ftrial_previous = Fval_trial; % сохраняем значения с предыдущего шага
                mu_trial_pervious = mu_trial;
                mu_trial = mu_trial*alfa;
                x_trial= x - mu_trial*grad_direction; % рассчитываем координату для следующей пробной точки
                Fval_trial=F(x_trial);% рассчитываем значение функции для это пробной точки
                % флажок проверки сходимости, определяется изменением функции на 
                % последовательных итерациях
                
                if is_return_search_history
                    search_history(:,ii+jj+1) = x_trial;
                end
                jj=jj+1;
                if Fval_trial<Ftrial_previous % произошло уменьшение 

                else % произошло увеличение 
                   mu_trial=mu_trial_pervious*beta;
                   break
                end
        end
        mu=mu_trial;
        x= x - mu*grad_direction;
        Fval=F(x);ii=ii+1;
        flag = [norm(Fval-F_previous)>tol ...
                    norm(x_previous-x)>tol ...
                    grad_norm>tol]; 
        
    end
    if is_return_search_history
        search_history = search_history(:,1:ii);
    end
end
