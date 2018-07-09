
clear;
rng(100);

% 读取二分类数据，乳腺癌病人数据集
data = csvread('./data/data.csv');
X = data(:, 1:end-1); size(X)
y = data(:, end); size(y)

% 测试sigmoid函数
% a = rand(5, 3);
% sigmoid(a)

% 进行FM SGD模型测试
alpha_w = 0.04;
beta_w = 0.05;
alpha_v = 0.03;
beta_v = 0.06;
sig_init = 0.3;
iter_num = 10;
l1_w = 0.003;
l2_w = 0.001;
l1_v = 0.003;
l2_v = 0.002;
k = 5;
[w0, w, v, min_x, max_x, loss] = fm_ftrl_train(X, y, alpha_w, beta_w, k, ...
    alpha_v, beta_v, l1_w, l2_w, l1_v, l2_v, sig_init, iter_num);
w = [w0; w];
[ a ] = fm_predict(X, w, v, min_x, max_x);

% accuracy
ac = a >= 0.5;
acc = sum(ac == y) / size(y, 1);
fprintf('FM FTRL train accuracy is: %.3f\n', acc)

% auc
[Xlog,Ylog,Tlog,AUClog] = perfcurve(y,a,1);
plot(Xlog, Ylog)
hold on 
legend(sprintf('Factor Machine with FTRL Classifier with AUC: %.3f', AUClog))
xlabel('False positive rate')
ylabel('True positive rate')
title('Performance of FM')
hold off

% fit linear
% rng(1); % For reproducibility
% [Mdl,FitInfo] = fitclinear(X,y);
% y_hat = predict(Mdl, X);
% fprintf('SVM train classifier accuracy is: %.3f\n', sum(y_hat == y) / size(y, 1))
