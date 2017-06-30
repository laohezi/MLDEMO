#将文字转化为词组向量
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)

content = ["how to format my hard disk","Hard disk format problems"]

X = vectorizer.fit_transform(content)
print(vectorizer.get_feature_names())#获取特征
print(X.toarray().transpose())#





