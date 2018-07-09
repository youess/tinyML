
%
% 计算sigmoid激荡函数
%

function [y] = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end