# -*- coding: utf-8 -*-

"""
计算AUC相当于计算模型排序一个随机正样本的概率比随机一个负样本高
Proof:
    https://kevincodeidea.wordpress.com/2017/01/23/proof-of-the-auc-is-equivalent-to-probability-of-ranking-positives-over-negatives/
"""
import pandas as pd


def cal_auc(y_real, y_preds):
    
    n = len(y_real)
    y = zip(y_real, y_preds)
    # 按照正样本预测概率高到低进行排序
    y = sorted(y, key=lambda k: -k[1])
    pos, neg = 0, 0
    pos_rnk = 0
    for i in range(n):
        if y[i][0] == 1:
            pos += 1
            pos_rnk += n - i
        else:
            neg += 1

    numerator = pos_rnk - 0.5 * pos * (pos + 1)
    denominator = pos * neg
    return numerator / denominator


def main():
    data = pd.read_csv("./classprobabilities.csv")
    auc_score = cal_auc(data.iloc[:, 1].tolist(), data.iloc[:, 2].tolist())
    
    print("auc score is: %.4f" % auc_score)  # 0.9556

if __name__ == "__main__":
    main()
