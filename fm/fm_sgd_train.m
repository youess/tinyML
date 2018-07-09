%
% FM SGD优化训练模型
%
function [w, v, min_x, max_x, loss] = fm_sgd_train(X, y, step_size, l1_w, l2_w, k, sig_init, l1_v, l2_v, iter_num)
    % 样本，维度数目
    m = size(X, 1); n = size(X, 2);
    % 初始化原始权重，以及因子权重
    w = zeros(n+1, 1); 
    v = normrnd(0, sig_init, n, k);  % v: 31 x 100
    % 计算数据特征最小，最大值进行归一化
    min_x = min(X, [], 1);      % same as min(X), 计算特征最小值
    max_x = max(X, [], 1);
    X_norm = (X - min_x) ./ (max_x - min_x); % size(X_norm)
    % 添加截距
    X_norm = [ones([m 1]) X_norm];
    
    for ti=1:iter_num      % 迭代数目
        fprintf('Training iteration: %d ;', ti)
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
            a = sigmoid(z);
            
            % % 计算损失量
            yy = y(mi);
            loss = -yy .* log(a) - (1 - yy) .* log(1 - a);
            
            % 更新梯度w
            dw = (a-yy) .* x;
            % % 添加正则项
            dw(2:end, :) = dw(2:end, :) + l1_w + 2 .* w(2:end, :) .* l2_w;
            w = w - step_size .* dw;
            
            % 更新梯度v
            for ki = 1:k
                dv = a - yy;
                dv = dv .* ( v(:, ki).' * x_ .* x_ - v(:, ki) .* x_.^2 );
                dv = dv + l1_v + v(:, ki) .* l2_v;
                v(:, ki) = v(:, ki) - step_size .* dv;
            end
            
        end
        fprintf(' Training loss : %.3f\n', loss)
    end
end
