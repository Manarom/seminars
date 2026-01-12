function D = DiffMat2(rows,cols)
% M- number of input matrix rows
    persistent Di 
    if isempty(Di) || rows~=size(Di,1) || cols ~= size(Di,2)
        Di = zeros(rows,cols);
        Di(diagonal_inds(1,rows,cols)) = -1;
        Di(diagonal_inds(-1,rows,cols)) = 1;
    end
    D = Di;
end