classdef Array < Collection 
    properties (SetAccess=protected) %(SetAccess = private)
        array  %double {mustBePositive}
    end
    methods
        function obj = Array(array)
            obj.array = array;
        end

        function out = get_index(obj,varargin)
            out = obj.array(varargin{:});
        end
        function out = length(obj)
            out= length(obj.array);
        end
        function set_index(obj,i,v)
            if i>length(obj) || i<=0
                error("Out of bounds")
            end
            obj.array(i) =v;
        end
    end
    
end
%  matlab.mixin.indexing.RedefinesParen
%       cat             - Concatenate objects
%       empty           - Create empty object of this class
%       parenAssign     - Overload parentheses indexed assignment
%       parenDelete     - Overload parentheses indexed deletion
%       parenListLength - Length of comma-separated list for parentheses
%                         indexing
%       parenReference  - Overload parentheses indexed reference
%       size            - Size of the object