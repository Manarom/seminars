classdef IndexingType<ArrayInd
    % тестовый класс, чтобы посмотреть различные типы индексирования 
    methods (Access = public) function obj= IndexingType();   obj@ArrayInd([]); end end
    % тут важный момент, атрибуты методов не должны сужать 
        % область их вивдимости по сравнению с предком, то есть,
        % если у класса предка имеется абстрактный метод с именем A, 
        % который помещен в блок c аттрибутом (Access = protected) мы 
        % должны будем конкретную реализацию этого метода также
        % делать с темже атрибутом
    methods (Access = protected)  function i = parenReference(obj,i); end   end
end