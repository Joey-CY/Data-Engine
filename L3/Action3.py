import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv("./car_data.csv", encoding="gbk")
train_x = data[["人均GDP", "城镇人口比重", "交通工具消费价格指数", "百户拥有汽车量"]]
kmeans = KMeans(n_clusters=3, max_iter=300)
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
# 有的时候收到的csv文件的列名是中文的，想要修改列名就需要在字符串前面加上 u，否则更改不会生效
result.rename({0: u"聚类结果"}, axis=1, inplace=True)
result.to_csv("./car_cluster_result.csv", index=False)
