from ui_plotly_setting import Ui_Form
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog, QApplication, QColorDialog
from PyQt5.QtCore import pyqtSlot
from plotly_pyqt5 import Plotly_PyQt5
import sys
import numpy as np
import plotly.offline as pyof   #离线绘图模式
import plotly.graph_objs as go
class Plotly_Setting(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(Plotly_Setting, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("绘图对话框")
        self.plotly_charts = Plotly_PyQt5()
        self.data = []
        self.tt = 0

        self.hoverlabel_font_color = '#ffaa00'
        self.hoverlabel_bordercolor = '#ffaa00'
        self.hoverlabel_bgcolor = '#ffaa00'
        self.textfont_color = '#ffaa00'
        self.marker_color = '#ffaa00'
        self.line_color = '#ffaa00'
        self.layout_title_font_color = '#ffaa00'
        #self.layout_title_pad = '#ffaa00'
    #添加子图数据
    def add_data(self, name, mode, marker_size, marker_color, line_width, line_color, opacity, x0, y0, dx, dy, text, textposition,
                 hoverinfo, hoveron, fill, line_dash,
                 hoverlabel_bgcolor, hoverlabel_bordercolor, hoverlabel_font_family, hoverlabel_font_size, hoverlabel_font_color,
                 connectgaps,  xcalendar, ycalendar, marker_opacity):
        self.tt =self.tt + 1

        N = 100
        random_x = np.linspace(0, 1, N)
        random_y0 = np.random.randn(N) + 5
        random_y1 = np.random.randn(N)
        random_y2 = np.random.randn(N) - 5

        # Create traces
        #data
        trace0 = go.Scatter(
            x=random_x,
            y=random_y0,
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            text=text,
            mode=mode,  # 样式
            name=name,
            connectgaps=connectgaps,
            #showscale=showscale,
            textposition=textposition,
            opacity=opacity,

            #hovertext=hovertext,
            hoverinfo=hoverinfo,
            hoveron=hoveron,
            fill=fill,
            ycalendar=ycalendar,
            xcalendar=xcalendar,
            #type=type,

            hoverlabel=dict(
                bgcolor=hoverlabel_bgcolor,
                bordercolor=hoverlabel_bordercolor,
                font=dict(
                    family=hoverlabel_font_family,
                    size=hoverlabel_font_size,
                    color=hoverlabel_font_color
                )
            ),
            line=dict(
                width=line_width,  # 设置线条宽度
                color=line_color,  # 设置线条的颜色
                # line=line_shape,
                dash=line_dash
            ),
            marker=dict(
                size=marker_size,  # 设置点的宽度
                color=marker_color,  # 设置点的颜色
                opacity=marker_opacity,
            ),
            selected=dict(
                marker=dict(
                    opacity=0.9,
                    size=20
                )
            )
        )

        self.data.append(trace0)


    def add_layout(self, layout_title_text, layout_title_font_family, layout_title_font_size, layout_title_font_color, layout_title_x,
                   layout_title_y, layout_title_xanchor, layout_title_yanchor, layout_title_pad):
        self.layout = dict(
            title=dict(text=layout_title_text,
                       font=dict(
                           family=layout_title_font_family,
                           size=layout_title_font_size,
                           color=layout_title_font_color,
                       ),
                       x=layout_title_x,
                       y=layout_title_y,
                       xanchor=layout_title_xanchor,
                       yanchor=layout_title_yanchor,
                       pad=dict(
                         b=5,
                         l=10,
                         r=3,
                         t=5,
                       ),
                       ),
            xaxis=dict(title='Date'),  # 横坐标名称
            yaxis=dict(title='Price'),  # 横坐标名称
        )
    #显示
    @pyqtSlot()
    def on_confirm_Btn_clicked(self):

        self.plotly_charts.scatter(data=self.data, layout=self.layout)
    #关闭
    @pyqtSlot()
    def on_cancel_Btn_clicked(self):
        self.close()
    #添加颜色
    @pyqtSlot()
    def on_line_color_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.line_color = color.name()
            self.label_28.setStyleSheet('background-color:%s' % self.line_color)
    #添加颜色
    @pyqtSlot()
    def on_marker_color_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.marker_color = color.name()
            self.label_2.setStyleSheet('background-color:%s' % self.marker_color)
    @pyqtSlot()
    def on_textfont_color_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.textfont_color = color.name()
            self.label_13.setStyleSheet('background-color:%s' % self.textfont_color)

    #hoverlabel
    @pyqtSlot()
    def on_hoverlabel_bgcolor_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.hoverlabel_bgcolor = color.name()
            self.label_17.setStyleSheet('background-color:%s' % self.hoverlabel_bgcolor)

    @pyqtSlot()
    def on_hoverlabel_bordercolor_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.hoverlabel_bordercolor = color.name()
            self.label_18.setStyleSheet('background-color:%s' % self.hoverlabel_bordercolor)

    @pyqtSlot()
    def on_hoverlabel_font_color_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.hoverlabel_font_color = color.name()
            self.label_25.setStyleSheet('background-color:%s' % self.hoverlabel_font_color)

    @pyqtSlot()
    def on_layout_title_font_color_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.layout_title_font_color = color.name()
            self.label_33.setStyleSheet('background-color:%s' % self.layout_title_font_color)
    '''
    @pyqtSlot()
    def on_layout_title_pad_Btn_clicked(self):

        color = QColorDialog.getColor()
        if color.isValid():
            self.layout_title_pad = color.name()
            self.label_3.setStyleSheet('background-color:%s' % self.layout_title_pad)
    '''
    #添加子图
    @pyqtSlot()
    def on_add_charts_Btn_clicked(self):
        ##########################################################data###########################################################
        #meta, customdata, xaxis, yaxis,orientation, groupnorm, stackgroup, marker.symbol, line.cauto, line.cmin等若干,namelength
        name = self.lineEdit_line_name.text()
        mode_index = self.comboBox_mode.currentIndex()
        self.mode = ''
        if mode_index == 0:
            self.mode = 'lines'
        if mode_index == 1:
            self.mode = 'markers'
        if mode_index == 2:
            self.mode = 'markers+lines'
        if mode_index == 3:
            self.mode = 'markers+lines+text'
        if mode_index == 4:
            self.mode = 'none'
        x0 = self.doubleSpinBox_x0.value()
        y0 = self.doubleSpinBox_y0.value()
        dx = self.doubleSpinBox_dx.value()
        dy = self.doubleSpinBox_dy.value()
        opacity = self.doubleSpinBox_opacity.value()
        marker_size = self.doubleSpinBox_marker_size.value()
        line_width = self.doubleSpinBox_line_width.value()
        text = self.lineEdit_text.text()
        textposition = self.comboBox_textposition.currentText()
        hoverinfo = self.comboBox_hoverinfo.currentText()
        hoveron = self.comboBox_hoveron.currentText()
        fill = self.comboBox_fill.currentText()
        line_dash = self.comboBox_line_dash.currentText()

        hoverlabel_font_family = self.comboBox_hoverlabel_font_family.currentText()
        hoverlabel_font_size = self.spinBox_hoverlabel_font_size.value()
        connectgaps = False
        if self.checkBox_scatter_connectgaps.isChecked():
            connectgaps = True
        xcalendar = self.comboBox_xcalendar.currentText()
        ycalendar = self.comboBox_ycalendar.currentText()

        marker_opacity = self.doubleSpinBox_marker_opacity.value()
        mode = self.mode
        marker_color = self.marker_color
        line_color = self.line_color
        textfont_color = self.textfont_color
        hoverlabel_bgcolor = self.hoverlabel_bgcolor
        hoverlabel_bordercolor = self.hoverlabel_bordercolor
        hoverlabel_font_color = self.hoverlabel_font_color
        self.add_data(name=name, mode=mode, marker_size=marker_size, marker_color=marker_color,
                      line_width=line_width, line_color=line_color, opacity=opacity, x0=x0, y0=y0,
                      dx=dx, dy=dy, text=text, textposition=textposition, hoverinfo=hoverinfo,
                      hoveron=hoveron, fill=fill, line_dash=line_dash, hoverlabel_bgcolor=hoverlabel_bgcolor,
                      hoverlabel_bordercolor=hoverlabel_bordercolor, hoverlabel_font_family=hoverlabel_font_family,
                      hoverlabel_font_size=hoverlabel_font_size, hoverlabel_font_color=hoverlabel_font_color, connectgaps=connectgaps,
                       xcalendar=xcalendar, ycalendar=ycalendar, marker_opacity=marker_opacity)


        ####################################################layout############################################################
        layout_title_text = self.lineEdit_layout_title_text.text()
        layout_title_font_family = self.comboBox_layout_title_font_family.currentText()
        layout_title_font_size = self.spinBox_layout_title_font_size.value()
        layout_title_font_color = self.layout_title_font_color
        layout_title_x = self.doubleSpinBox_layout_title_x.value()
        layout_title_y = self.doubleSpinBox_layout_title_y.value()
        layout_title_xanchor = self.comboBox_layout_title_xanchor.currentText()
        layout_title_yanchor = self.comboBox_layout_title_yanchor.currentText()
        layout_title_pad = self.comboBox_layout_title_pad.currentText()

        self.add_layout(layout_title_text=layout_title_text, layout_title_font_family=layout_title_font_family,
                        layout_title_font_size=layout_title_font_size, layout_title_font_color=layout_title_font_color,
                        layout_title_x=layout_title_x, layout_title_y=layout_title_y, layout_title_xanchor=layout_title_xanchor,
                        layout_title_yanchor=layout_title_yanchor, layout_title_pad=layout_title_pad)



        print("OK")

    #预览
    @pyqtSlot()
    def on_preview_Btn_clicked(self):
        '''
        name = self.lineEdit_line_name.text()
        mode_index = self.comboBox_mode.currentIndex()
        self.mode = ''
        if mode_index == 0:
            self.mode = 'lines'
        if mode_index == 1:
            self.mode = 'markers'
        if mode_index == 2:
            self.mode = 'markers+lines'
        if mode_index == 3:
            self.mode = 'markers+lines+text'
        if mode_index == 4:
            self.mode = 'none'
        x0 = self.doubleSpinBox_x0.value()
        y0 = self.doubleSpinBox_y0.value()
        dx = self.doubleSpinBox_dx.value()
        dy = self.doubleSpinBox_dy.value()
        opacity = self.doubleSpinBox_opacity.value()
        marker_size = self.doubleSpinBox_marker_size.value()
        line_width = self.doubleSpinBox_line_width.value()
        text = self.lineEdit_text.text()
        textposition = self.comboBox_textposition.currentText()
        #hovertext = self.lineEdit_text.text()
        hoverinfo = self.comboBox_hoverinfo.currentText()
        hoveron = self.comboBox_hoveron.currentText()
        fill = self.comboBox_fill.currentText()
        #line_shape = self.comboBox_line_shape.currentText()
        line_dash = self.comboBox_line_dash.currentText()
        #textfont_family = self.comboBox_textfont_family.currentText()
        #textfont_size = self.spinBox_textfont_size.value()
        hoverlabel_font_family = self.comboBox_hoverlabel_font_family.currentText()
        hoverlabel_font_size = self.spinBox_hoverlabel_font_size.value()
        connectgaps = False
        if self.checkBox_scatter_connectgaps.isChecked():
            connectgaps = True
        #showscale = True
        #if self.checkBox_showscale.isChecked():
            #showscale = True
        xcalendar = self.comboBox_xcalendar.currentText()
        ycalendar = self.comboBox_ycalendar.currentText()
        #type = self.comboBox_type.currentText()
        marker_opacity = self.doubleSpinBox_marker_opacity.value()
        mode = self.mode
        marker_color = self.marker_color
        line_color = self.line_color
        textfont_color = self.textfont_color
        hoverlabel_bgcolor = self.hoverlabel_bgcolor
        hoverlabel_bordercolor = self.hoverlabel_bordercolor
        hoverlabel_font_color = self.hoverlabel_font_color

        N = 100
        random_x = np.linspace(0, 1, N)
        random_y0 = np.random.randn(N) + 5
        random_y1 = np.random.randn(N)
        random_y2 = np.random.randn(N) - 5

        # Create traces
        trace0 = go.Scatter(
            x=random_x,
            y=random_y0,
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            text=text,
            mode=mode,  # 样式
            name=name,
            connectgaps=connectgaps,
            #showscale=showscale,

            textposition=textposition,
            opacity=opacity,

            #hovertext=hovertext,
            hoverinfo=hoverinfo,
            hoveron=hoveron,
            fill=fill,
            ycalendar=ycalendar,
            xcalendar=xcalendar,
            # type=type,

            hoverlabel=dict(
                bgcolor=hoverlabel_bgcolor,
                bordercolor=hoverlabel_bordercolor,
                font=dict(
                    family=hoverlabel_font_family,
                    size=hoverlabel_font_size,
                    color=hoverlabel_font_color
                )
            ),
            line=dict(
                width=line_width,  # 设置线条宽度
                color=line_color,  # 设置线条的颜色
                # line=line_shape,
                dash=line_dash,
            ),
            marker=dict(
                size=marker_size,  # 设置点的宽度
                color=marker_color,  # 设置点的颜色
                opacity=marker_opacity,

            )
        )



        ###############################################################layout################################################
        layout_title_text = self.lineEdit_layout_title_text.text()
        layout_title_font_family = self.comboBox_layout_title_font_family.currentText()
        layout_title_font_size = self.spinBox_layout_title_font_size.value()
        layout_title_font_color = self.layout_title_font_color
        self.add_layout(layout_title_text=layout_title_text, layout_title_font_family=layout_title_font_family,
                        layout_title_font_size=layout_title_font_size, layout_title_font_color=layout_title_font_color,
                        )
        layout = dict(
            title=dict(text=layout_title_text,
                       font=dict(
                           family=layout_title_font_family,
                           size=layout_title_font_size,
                           color=layout_title_font_color
                       )),
            xaxis=dict(title='Date'),  # 横坐标名称
            yaxis=dict(title='Price'),  # 横坐标名称
        )
        '''

        self.plotly_charts.scatter(data=[trace0], layout=layout)
if __name__=="__main__":
    app = QApplication(sys.argv)
    plotly_show = Plotly_Setting()
    plotly_show.show()
    #plotly_win = Plotly_Setting()
    #plotly_win.open()

    #plotly_show.plotly_setting()
    sys.exit(app.exec_())
