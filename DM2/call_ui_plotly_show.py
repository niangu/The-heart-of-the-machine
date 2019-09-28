import sys
from ui_plotly_show import Ui_Form
from call_plotly_setting import Plotly_Setting
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import pyqtSlot, QRect
from plotly_pyqt5 import Plotly_PyQt5
class Plotly_Show(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(Plotly_Show, self).__init__(parent)
        self.setupUi(self)
        self.plotly_pyqt5 = Plotly_PyQt5()
        #self.widget.setGeometry(QRect(50, 20, 1200, 600))
        #self.Form_2.load(QUrl.fromLocalFile(self.plotly_pyqt5.get_plotly_path_if_hs300_bais()))
        self.Form_2.load(QUrl('file:///ender.html'))
        #self.widget_2.load(QUrl.fromLocalFile(self.plotly_pyqt5.get_plot_path_matplotlib_plotly()))
        #self.plotly_pyqt5.get_plotly_path_if_hs300_bais()
        self.setWindowTitle("绘图")
#       self.plotly_pyqt5.scatter()
    def plotly_setting(self):
        plotly_win = Plotly_Setting()
        plotly_win.exec()

if __name__=="__main__":
    app = QApplication(sys.argv)
    plotly_show = Plotly_Show()
    plotly_show.show()
    #ttt = Plotly_Setting()
    #ttt.show()
    #plotly_show.plotly_setting()
    sys.exit(app.exec_())
