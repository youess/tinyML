
function [ a ] = fm_predict(X, w, v, min_x, max_x)
    
    m = size(X, 1);
    X_norm = (X - min_x) ./ (max_x - min_x);
    X_norm = [ones([m 1]) X_norm];
    a = zeros(m, 1);
    for mi=1:m         % 对于每个样本进行SGD优化
            % 预测类别变量a
            % % 先计算交互项部分 - (v*x)^2
            x = X_norm(mi, :).';         % 转置成特征向量32 x 1
            x_ = x(2:end, :);            % 没有截距的样本31 x 1
            inter1 = v.' * x_;  % 100 x 1
            inter2 = (v.^2).' * x_.^2;   
            interaction = 0.5 * sum(inter1.^2 - inter2);
            
            % % 计算预测类别
            z = w.' * x + interaction;
            a(mi) = sigmoid(z);
    end
end