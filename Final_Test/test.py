import pandas as pd


def project_a():
    import requests
    from bs4 import BeautifulSoup      
    url = "http://car.bitauto.com/xuanchegongju/?l=8&mid=8"
    headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    html=requests.get(url,headers=headers,timeout=10)
    content = html.text
    soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')
    tags = soup.find_all("div", class_="search-result-list-item")
    df = pd.DataFrame(columns=["name", "l_price", "h_price", "pic"])
    name = []
    pic = []
    price_min = []
    price_max = []
    for tag in tags:
        pic.append(tag.a.img["src"])
        price_tags = tag.a.find_all("p", class_="cx-price")
        name_tags = tag.a.find_all("p", class_="cx-name text-hover")
        for price_tag in price_tags:
            price_tag_range = price_tag.get_text()
            price_tag_range = price_tag_range.replace("万", "").split("-")
            price_tag_min = price_tag_range[0]
            if len(price_tag_range) == 2:
                price_tag_max = price_tag_range[1]
            elif len(price_tag_range) == 1:
                price_tag_max = price_tag_range[0]
            price_min.append(price_tag_min)
            price_max.append(price_tag_max)
            #df = df.groupby([name, price_min, price_max, pic], axis=0)
        temp = pd.Series([name_tags[0].get_text(), price_tag_min, price_tag_max, tag.a.img["src"]], index=["name", "l_price", "h_price", "pic"])
        df = df.append(temp, ignore_index=True)
    df.to_csv("carprice.csv", index=False)        


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
    
def project_b_rule1():
    from efficient_apriori import apriori    
    data = pd.read_csv("./订单表.csv", encoding="gbk")
    orders_series = data.set_index("客户ID")["产品名称"]
    orders_series.sort_index(inplace=True)
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
    itemsets, rules = apriori(transactions, min_support=0.05, min_confidence=0.5)
    print("频繁项集", itemsets)
    print("关联规则", rules)
            
        
def project_b_rule2():
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules    
    data = pd.read_csv("./订单表.csv", encoding="gbk")
    hot_encoded_df = data.groupby(["客户ID", "产品名称"])["产品名称"].count().unstack().reset_index().fillna(0).set_index("客户ID")
    hot_encoded_df = hot_encoded_df.applymap(encode_units)
    frequent_itemsets = apriori(hot_encoded_df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
    print("频繁项集", frequent_itemsets)
    print("关联规则", rules[(rules["lift"]>=1) & (rules["confidence"]>=0.5)])


def project_c():
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder        
    data = pd.read_csv("./CarPrice_Assignment.csv")
    train_x = data.drop(["car_ID", "CarName"], axis=1)
    le = LabelEncoder()
    temps = ["fueltype", "aspiration", "doornumber", "carbody", "drivewheel", 
             "enginelocation", "enginetype", "cylindernumber", "fuelsystem"]
    for temp in temps:        
        train_x[temp] = le.fit_transform(train_x[temp])
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x = min_max_scaler.fit_transform(train_x)
    #print(train_x)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train_x)
    predict_y = kmeans.predict(train_x)
    result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
    result.rename({0:"聚类结果"}, axis=1, inplace=True)
    result.to_csv("car_cluster_result.csv", index=False)
    
    
#project_a()
project_b_rule1()
project_b_rule2()
project_c()