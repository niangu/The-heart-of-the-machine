import pandas as pd
import numpy as np

df = pd.read_csv('/home/niangu/桌面/比赛文件/文件1.csv')
sepch = df['GPS车速']
a = []
b = []
i = 0
c = []
f = []
#提取运动学片段
for index in range(len(sepch)):
    if sepch[index] < 10:
        c.append(index)
        b.append(sepch[index])
    else:
        lem_b = len(b)
        if lem_b > 0:
            if lem_b < 180:
                f.append(c)
                a.append(b)
                b = []
                c = []
            else:
                b = []
                c = []

#pd.Series(b).to_csv('/home/niangu/桌面/比赛文件/文件1(运动学片段).csv')
print()
print()
print(a)
print(len(a))
print(f)
print(len(f))


