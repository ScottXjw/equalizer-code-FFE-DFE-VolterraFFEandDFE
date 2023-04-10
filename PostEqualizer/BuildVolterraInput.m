function [res] = BuildVolterraInput(input_X,order)
% 构建 Volterra 输入
% input_X: 输入信号向量
% order: Volterra 级数
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







