% 尝试实现FM的FTRL优化算法的实现

function [w0, w, v, min_x, max_x, loss] = fm_ftrl_train(X, y, ...
    alpha_w, beta_w, k, ...
    alpha_v, beta_v, ...
    l1_w, l2_w, l1_v, l2_v, sig_init, iter_num)
    
    [m, n] = size(X);
    w0 = 0;
    nw0 = 0;
    zw0 = 0;
    
    w = zeros(n, 1);
    nw = zeros(n, 1);
    zw = zeros(n, 1);
    
    v = normrnd(0, sig_init, n, k);
    nv = zeros(n, k);
    zv = zeros(n, k);
    
    % 计算数据特征最小，最大值进行归一化
    min_x = min(X, [], 1);      % same as min(X), 计算特征最小值
    max_x = max(X, [], 1);
    X_norm = (X - min_x) ./ (max_x - min_x); % size(X_norm)
    % 添加截距
    X_norm = [ones([m 1]) X_norm];
    
    for ti = 1:iter_num
        fprintf('Iteration %d ', ti)
        fprintf(' v sum: %.3f ', sum(sum(v)))
        for mi = 1:m    
        
            % 接收特征向量
            x = X_norm(mi, :).';
            x_ = x(2:end);           
            
            % 计算预测类别
            inter1 = v.' * x_; 
            inter2 = (v.^2).' * x_.^2;   
            interaction = 0.5 * sum(inter1.^2 - inter2);
            
            % % 计算预测类别
            z = w0 .* x(1) + w.' * x_ + interaction;
            a = sigmoid(z);
            
            yy = y(mi);
            
            loss = -yy .* log(a) - (1 - yy) .* log(1 - a);
            
            % 计算权重梯度
            dw0 = (a - yy);
            sigma_w0 = ( sqrt(nw0 + dw0.^2) - sqrt(nw0) ) ./ alpha_w;
            zw0 = zw0 + dw0 - sigma_w0 .* w0;
            nw0 = nw0 + sigma_w0.^2;
            
            
            for xi = 1:n
                if xi ~= 0
                    
                    dwi = (a - yy) .* x(xi);
                    sigma_wi = ( sqrt(nw(xi) + dwi.^2) - sqrt(nw(xi)) ) ./ alpha_w;
                    zw(xi) = zw(xi) + dwi - sigma_wi .* dwi;
                    nw(xi) = nw(xi) + sigma_wi.^2;
                    
                    for ki = 1:k
                        dvi = a - yy;
                        dvi = dvi .* ( v(:, ki).' * x_ .* x_(xi) - v(xi, ki) .* x_(xi).^2 );
                        sigma_vi = ( sqrt(nv(xi, ki) + dvi.^2) - sqrt(nv(xi, ki)) ) ./ alpha_v;
                        zv(xi, ki) = zv(xi, ki) + dvi - sigma_vi .* v(xi, ki);
                        nv(xi, ki) = nv(xi, ki) + dvi.^2;
                    end
                    
                end
            end
            
            % 处理w0
            if abs(zw0) <= l1_w
                 w0 = 0;
            else
                 % 整理下计算逻辑
                 d1 = alpha_w .* l2_w + beta_w + sqrt(nw0);
                 d2 = alpha_w .* (zw0 - sign(zw0) .* l1_w);
                 w0 = -d2 ./ d1;
            end
            
            for xi = 1:n
                if xi ~= 0
                    
                    % 处理w权重
                    if abs(zw(xi)) <= l1_w
                        w(xi) = 0;
                    else
                        d1 = alpha_w .* l2_w + beta_w + sqrt(nw(xi));
                        d2 = alpha_w .* (zw(xi) - sign(zw(xi)) .* l1_w);
                        w(xi) = -d2 ./ d1;
                    end
                    
                    % 处理v，交叉变量权重
                    for ki = 1:k
                        if abs(zv(xi, ki)) <= l1_v              % 
                            v(xi, ki) = 0;
                        else
                            d1 = alpha_v .* l2_v + beta_v + sqrt(nv(xi, ki));
                            d2 = alpha_v .* (zv(xi, ki) - sign(zv(xi, ki)) .* l1_v);
                            v(xi, ki) = -d2 ./ d1;
                        end

                        % TODO: 是否强迫重新初始化，当wi为0时，v(xi, ki)就重新初始化, 提高稀疏性可以加

                    end 
                    
                end
            end

        end

        fprintf(' Loss %.3f\n', loss)
    
    end

end