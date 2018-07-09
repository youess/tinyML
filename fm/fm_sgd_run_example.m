
clear;
rng(101);

% 读取二分类数据，乳腺癌病人数据集
data = csvread('./data/data.csv');
X = data(:, 1:end-1); size(X)
y = data(:, end); size(y)

% 测试sigmoid函数
a = rand(5, 3);
sigmoid(a)

% 进行FM SGD模型测试
alpha = 0.1;
sig_init = 0.3;
iter_num = 100;
l1_w = 0.01;
l2_w = 0.01;
l1_v = 0.01;
l2_v = 0.01;
k = 5;
[w, v, min_x, max_x, loss] = fm_sgd_train(X, y, alpha, l1_w, l2_w, k, sig_init, l1_v, l2_v, iter_num);
[ a ] = fm_predict(X, w, v, min_x, max_x);

% accuracy
ac = a >= 0.5;
acc = sum(ac == y) / size(y, 1);
fprintf('FM SGD train accuracy is: %.3f\n', acc)

% auc
[Xlog,Ylog,Tlog,AUClog] = perfcurve(y,a,1);
plot(Xlog, Ylog)
hold on 
legend(sprintf('Factor Machine Classifier with AUC: %.3f', AUClog))
xlabel('False positive rate')
ylabel('True positive rate')
title('Performance of FM')
hold off

% fit linear
% rng(1); % For reproducibility
% [Mdl,FitInfo] = fitclinear(X,y);
% y_hat = predict(Mdl, X);
% fprintf('SVM train classifier accuracy is: %.3f\n', sum(y_hat == y) / size(y, 1))
