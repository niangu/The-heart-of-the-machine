'''
#保存加载模型
from __future__ import print_function
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

#method 1: pickle
import pickle
#save
with open('save/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
#restore
with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    print("pickle存储结果", clf2.predict(X[0, 1]))


#method 2: joblib
from sklearn.externals import joblib
#Save
joblib.dump(clf, 'save/clf.pk1')
#restore
clf3 = joblib.load('save/clf.pk1')
print(clf3.predict(X[0:1]))
'''


import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as pyof   #离线绘图模式
import plotly.graph_objs as go

#df = pd.read_csv(r'/home/niangu/桌面/数据挖掘与建模/creditcard.csv')

#plt.scatter(df['V1'], df['V2'])
#plt.show()
#print("OK")
'''
df = pd.read_csv(r'/home/niangu/桌面/数据挖掘与建模/creditcard.csv')
#df = pd.read_csv(r'/home/niangu/桌面/数据挖掘与建模/HR.csv')
#绘制散点图
#曲线1
line_main_price = go.Scatter(
    #x=df['V2'],         #x轴数据
    #y=df['Time'],#y轴数据数据
    #x=df.index,
    #y=df['left'],
    #mode='markers+lines',
 #   name='hs300_close',#名字
    #marker=dict(#
     #    size=2, #设置点的宽度
         #color='rgba(152, 0, 0, .8)',#设置点的颜色
      #   line=dict(
       #      width=2, #设置线条宽度
             #color='rgb(0, 0, 0)'#设置线条的颜色
         )
      #)
 #)
    #connectgaps=True,#允许连接数据缺口


#曲线2
#line_hs300_close = go.Scatter(
    #x=df['V2'],
   #  name='hs300_close',
 #   connectgaps=True,
#)

#data = [line_hs300_close, line_main_price]
data = [line_main_price]
layout = dict(title='if_hs300_bais',
              xaxis=dict(title='Date'),#横坐标名称
              yaxis=dict(zeroline=True),#显示y轴0刻度线
              #xaxis=dict(zeroline=False),#显示x轴0刻度线
              #yaxis=dict(title='Price'),#横坐标名称
              )
#fig = go.Figure(data=data, layout=layout)
#pyof.plot(fig, filename='text.html', auto_open=True) #自动在浏览器中打开设置为False
'''
'''
import pyecharts as pye

v1 = df['V1']
v2 = df['V2']
scatter = pye.Scatter()
scatter.add("", v1, v2)
scatter.render()
'''
'''
import pyqtgraph as pg
import pandas as pd
app = pg.mkQApp()
plt2 = pg.plot()
df = pd.read_csv(r'/home/niangu/桌面/数据挖掘与建模/creditcard.csv')
x = df['V1']
y = df['V2']
plt2.plot(x, y, pen=None, symbol="o")

app.exec()
'''
import pandas as pd
pd.set_option('display.max_columns', None)#显示所有列
df = pd.read_csv("/home/niangu/桌面/比赛文件/文件1.csv",sep=',')
'''
# #转换为时间序列
df = pd.read_csv('/home/niangu/桌面/比赛文件/文件1.csv', sep=',')
print("AAAAAAAAAAAAAAAAAAAAA")
df['时间'] = df['时间'].str.split('.000.')

print(df['时间'])
print("转换前：", df['时间'].dtypes)
df['时间'] = df['时间'].apply(lambda x:pd.to_datetime(x[0]))
print("转换后", df['时间'].dtypes)
print(df['时间'])
df.to_csv('/home/niangu/桌面/比赛文件/文件1.csv')
'''
'''
#print(df.describe())
import numpy as np
row, col = df.shape
df = pd.read_csv('/home/niangu/桌面/比赛文件/文件1.csv', sep=',')
print("AAAAAAAAAAAAAAAAAAAAA")
df['时间'] = df['时间'].str.split('.000.')

print(df['时间'])
print("转换前：", df['时间'].dtypes)
df['时间'] = df['时间'].apply(lambda x:pd.to_datetime(x[0]))
print("转换后", df['时间'].dtypes)

a = df['时间'].diff(1)#一阶拆分
df.insert(1, '时间差', a)
df.to_csv("/home/niangu/桌面/比赛文件/文件1.csv")

print(a)
print("AAAAAAAAAAA")
'''
#print(df['时间'])
print("转换前：", df['时间'].dtypes)
df['时间'] = df['时间'].apply(lambda x: pd.to_datetime(x))
#df['时间差'] = df['时间差'].apply(lambda x: pd.to_datetime(x))
print("转换后", df['时间'].dtypes)
a = df['时间'].diff(1)
print(a)
df.to_csv('/home/niangu/桌面/比赛文件/文件1.csv')
print("AAAAAAAAAA")
