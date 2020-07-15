# encoding=utf-8
import pandas as pd


def data_process():
    data = pd.read_csv("./Market_Basket_Optimisation.csv", header=None)
    transactions = []
    for i in range(data.shape[0]):
        temp = []
        for j in range(data.shape[1]):
            if str(data.values[i, j]) != "nan":
                temp.append(str(data.values[i, j]))
        transactions.append(temp)
    return transactions


def main():
    transactions = data_process()
    from efficient_apriori import apriori
    itemsets, rules = apriori(transactions, min_support=0.05, min_confidence=0.15)
    print("频繁项集: ", itemsets)
    print("关联规则: ", rules)


main()
