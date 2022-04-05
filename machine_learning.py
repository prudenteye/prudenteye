#熟悉sklearn
'''
#引入一个iris花的数据集，根据其大小参数自动判断花的种类
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_x,iris_y,test_size=0.3)
print(iris)

knn = KNeighborsClassifier()
knn.fit(x_train,y_train)

print(knn.predict(x_test))

print(y_test)
'''

'''
#构造一个线性回归的数据级，然后通过matplotlib绘制出来
from sklearn import datasets

x,y = datasets.make_regression(n_samples=150, n_features=2,n_targets=2,noise=100)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(x,y)
plt.show()
'''

#分割文本并向量化
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import jieba

vector = CountVectorizer()

c1 = jieba.cut("男性，身高一米八且短头发，喜欢打球")
c2 = jieba.cut("男性，身高一米七又短头发，喜欢打代码")
c3 = jieba.cut("女性，身高一米六还长头发，喜欢逛淘宝")
c4 = jieba.cut("女性，身高一米五五没错短头发，喜欢写公众号")

str1 = "".join(list(c1))
str2 = "".join(list(c2))
str3 = "".join(list(c3))
str4 = "".join(list(c4))

res = vector.fit_transform([str1,str2,str3,str4])

print(vector.get_feature_names())

print(res.toarray())
