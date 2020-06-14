# Action1.1: 求2+4+6+8+...+100的求和，用Python该如何写
i = 0
a = 0
while i <= 100:
    a += i
    i += 2
print(a)

# Action1.2: 统计全班的成绩
import pandas as pd
from pandas import DataFrame, Series
data = pd.read_excel('data.xlsx')
data.describe()
data["sum"] = data["语文"] + data["数学"] + data["英语"]
data = data.sort_values("sum")
data.index = range(1, 6)
print(data)
data.to_excel('data1.xlsx')

# Action1.3: 对汽车质量数据进行统计
import pandas as pd
result = pd.read_csv("car_complain.csv")
result = result.drop("problem", 1).join(result.problem.str.get_dummies(","))
tags = result.columns[7:]
# 按照brand统计投诉总数，不同problem类型的总数
df1 = result.groupby(["brand"])["id"].agg(["count"])
df2 = result.groupby(["brand"])[tags].agg(["sum"])
# 按照投诉总数进行排序
df3 = pd.merge(df1, df2, on="brand")
df3.reset_index(inplace=True)
df3 = df3.sort_values("count", ascending=False)
query = ("Q350", "sum")
# 按照指定的problem类型进行排序
df3.sort_values(query, ascending=False)
