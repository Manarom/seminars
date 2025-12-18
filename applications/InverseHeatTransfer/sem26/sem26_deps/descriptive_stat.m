function stat = descriptive_stat(options)
    arguments
        options.y
        options.y_hat
        options.p
        options.name string {mustBeMember(options.name ,["all", "SSE","SST","r2","r2adj","AIC","BIC","AICC","RIC"])} = "all"
    end
    names = ["SSE","SST","r2","r2adj"; ...
              "AIC","BIC","AICC","RIC"];
    y = options.y;y_hat = options.y_hat;p = options.p;

    
    stat_crit_names = cellstr(names);
    vals = zeros(2,4);
    s = sig2(y,y_hat,p);
    n  = size(y_hat,1);
    args = {y,y_hat,p;...
            n,p,s};
    for jj = 1:2
        args_cur = args(jj,:);
        for ii = 1:size(stat_crit_names,2)
            vals(jj,ii) = feval(stat_crit_names{jj,ii},args_cur{:});
        end
    end
    stat_args = cell(2*numel(stat_crit_names),1);
    stat_args(1:2:end) = stat_crit_names;
    stat_args(2:2:end) = num2cell(vals);
    stat_args = transpose(stat_args);
    stat = struct(stat_args{:});
end

function v = AIC(n,p,s)
    v = n*log(s) + 2*p;
end
function v = BIC(n,p,s)
    v = n*log(s) + p*log(n);
end
function v = AICC(n,p,s)
    v = n*log(s) + 2*p.*(p+1)./(n - p - 1);
end
function v = RIC(n,p,s)
    v = (n-p)*log(s) + p*(log(n)-1)  + 4./(n - p - 2);
end
function v = SSE(y,y_hat,~)
    v = sumsqr(y - y_hat);
end
function v = SST(y,~,~)
    v = sumsqr(y - mean(y));
end
function v = r2(y,y_hat,~)
    v = 1 - SSE(y,y_hat)./SST(y,y_hat);
end
function v = r2adj(y,y_hat,p)
    n = length(y);
    v = 1 - r2(y,y_hat)*(n - 1)/(n - p);
end
function v = sig2(y,y_hat,p)
    n = length(y);
    v = SSE(y,y_hat)/(n - p);
end