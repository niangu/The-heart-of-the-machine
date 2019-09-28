import pandas as pd
df = pd.read_csv('/home/niangu/桌面/比赛文件/文件1.csv')
print(df['时间'].dtype)
