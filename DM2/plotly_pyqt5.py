import pandas as pd
import os
import plotly.offline as pyof   #离线绘图模式
import plotly.graph_objs as go

import numpy as np
import matplotlib.pyplot as plt

class Plotly_PyQt5():
    def __init__(self):
        '''初始化时设置存储HTML文件的文件夹名称,默认为plotly_html'''
        plotly_dir = 'plotly_html' #设置存储文件夹名称
        if not os.path.isdir(plotly_dir):
            os.mkdir(plotly_dir)

        self.path_dir_plotly_html = os.getcwd() + os.sep + plotly_dir

    def get_plotly_path_if_hs300_bais(self, file_name='if_hs300_bais.html'):
        '''
        path_plotly = self.path_dir_plotly_html + os.sep + file_name
        #df = pd.read_excel(r'data/if_index_bais.xlsx')
        df = pd.read_csv(r'/home/niangu/桌面/数据挖掘与建模/creditcard.csv')
        #df = pd.read_csv(r'/home/niangu/桌面/数据挖掘与建模/HR.csv')
        #绘制散点图
        #曲线1
        line_main_price = go.Scatter(
            x=df['V2'],         #x轴数据
            y=df['Time'],#y轴数据数据
            mode='markers+lines',
            name='hs300_close',#名字
            marker=dict(
                 #size=10, #设置点的宽度
                 #color='rgba(152, 0, 0, .8)',#设置点的颜色
                 #line=dict(
                     #width=2, #设置线条宽度
                     #color='rgb(0, 0, 0)'#设置线条的颜色
                 #)
             )
         )
            #connectgaps=True,#允许连接数据缺口


        #曲线2
        line_hs300_close = go.Scatter(
            #x=df.index,
            #y=df['hs300_close'],
            name='hs300_close',
            connectgaps=True,
        )

        #data = [line_hs300_close, line_main_price]
        data = [line_main_price]
        layout = dict(title='if_hs300_bais',
                      xaxis=dict(title='Date'),#横坐标名称
                      yaxis=dict(zeroline=True),#显示y轴0刻度线
                      #xaxis=dict(zeroline=False),#显示x轴0刻度线
                      #yaxis=dict(title='Price'),#横坐标名称
                      )
        fig = go.Figure(data=data, layout=layout)
        pyof.plot(fig, filename=path_plotly, auto_open=True) #自动在浏览器中打开设置为False
        return path_plotly
        '''

    #def scatter(self, name, mode, marker_size, marker_color, line_width, line_color):
    def scatter(self, data, layout):

         fig = go.Figure(data=data, layout=layout)

         pyof.plot(fig, filename='plotly_html/scatter.html')

    '''
    def shiop(self):
        import pyecharts as pye
        df = pd.read_csv(r'/home/niangu/桌面/数据挖掘与建模/creditcard.csv')
        v1 = df['V1']
        v2 = df['V2']
        scatter = pye.Scatter()
        scatter.add("", v1, v2)
        scatter.render()
    '''