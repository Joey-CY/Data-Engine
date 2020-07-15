# coding=utf-8
import pandas as pd
import time


def data_process():
    data = pd.read_csv("./BreadBasket_DMS.csv")
    data["Item"] = data["Item"].str.lower()
    data = data.drop(data[data.Item == "none"].index)
    return data

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def rule1():
    from efficient_apriori import apriori
    start = time.time()
    data = data_process()
    # 将"Transaction"这一列作为索引
    orders_series = data.set_index("Transaction")["Item"]
    transactions = []
    temp_index = 0
    for i, v in orders_series.items():
        if i != temp_index:
            temp_set = set()
            temp_index = i
            temp_set.add(v)
            transactions.append(temp_set)
        else:
            temp_set.add(v)
    itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.5)
    print('频繁项集：', itemsets)
    print('关联规则：', rules)
    end = time.time()
    print("用时：", end-start)


def rule2():
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    from mlxtend.preprocessing import TransactionEncoder
    data = data_process()
    pd.options.display.max_columns = 100
    start = time.time()
    # hot_encoded_df = data.groupby(['Transaction', 'Item'])['Item'].count().unstack().fillna(0) 该命令似乎也可以用
    hot_encoded_df = data.groupby(['Transaction', 'Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
    hot_encoded_df = hot_encoded_df.applymap(encode_units)
    frequent_itemsets = apriori(hot_encoded_df, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print("频繁项集：", frequent_itemsets)
    print("关联规则：", rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.5)])
    print(rules['confidence'])
    end = time.time()
    print("用时：", end-start)


def c_ncap():
    df = pd.read_csv("./ncap.csv", encoding="gbk")
    df = df.set_index(["车型"])
    #df.index = df["车型"]
    #df.drop(["车型"], axis=1, inplace=True)
    f = lambda x: 1 if x >=12 else 0
    df = df.applymap(f)

    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

    frequent_itemsets = apriori(df, min_support=0.02, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
    print("频繁项集：", frequent_itemsets)
    pd.options.display.max_columns = 100
    print("关联规则：", rules)


c_ncap()