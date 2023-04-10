function [res] = BuildVolterraInput(input_X,order)
% ���� Volterra ����
% input_X: �����ź�����
% order: Volterra ����
res = [];
if order == 2
    for i = 1:length(input_X)
        for j = i:length(input_X)
            res = [res input_X(i) * input_X(j)];
        end
    end
elseif order == 3
    for i = 1:length(input_X)
        for j = i:length(input_X)
            for m = j:length(input_X)
                res = [res input_X(i) * input_X(j) * input_X(m)];
            end
        end
    end
end
end







