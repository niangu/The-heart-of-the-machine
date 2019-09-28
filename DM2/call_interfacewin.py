import sys
from ui_interfacewin import Ui_MainWindow

from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog,QListWidgetItem,QMessageBox, QDialog
from PyQt5.QtCore import pyqtSlot,QModelIndex,Qt
from PyQt5.QtGui import QStandardItem,QStandardItemModel,QTextCursor,QTextOption
from qtpandas.models.DataFrameModel import DataFrameModel

from qtpandas.views.DataTableView import DataTableWidget
import pandas as pd
import scipy.stats as ss
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc, \
     roc_auc_score, accuracy_score, recall_score, f1_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.externals import joblib
import os
import pydotplus
import fpGrowth
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
os.environ["PATH"] += os.pathsep+"/home/niangu/桌面/DM2"
import math
np.set_printoptions(threshold=np.inf) #print全量数据
pd.set_option('display.max_columns', None)#显示所有列
pd.set_option('display.max_rows', None)#显示所有行
pd.set_option('max_colwidth', 100)#显示所有行
pd.set_option('display.width', 1000)
from scipy.interpolate import lagrange
from scipy.interpolate import interp1d
from scipy.interpolate import spline
import seaborn as sns
import matplotlib.pyplot as plt
from call_ui_plotly_show import *
from call_ui_savefile import SaveFile

class InterfaceWin(QMainWindow):
    def __init__(self, parent=None):
        super(InterfaceWin, self).__init__(parent)
        self.ui_interfacewin = Ui_MainWindow()
        self.ui_interfacewin.setupUi(self)
        #查看模型
        self.dataframemodel = DataFrameModel()
        self.ui_interfacewin.widget_datatablemodel.setViewModel(self.dataframemodel)
        self.ui_interfacewin.widget_datatablemodel.setButtonsVisible(False)
        #分析模型
        self.standardmodel = QStandardItemModel()
        self.ui_interfacewin.tableView_standardmodel.setModel(self.standardmodel)
        #显示图表
        #self.plotly_show = Plotly_Show()
        #self.plotly_show.show()
        self.Call_ui_savefile = SaveFile()
       # self.plotly_show.hide()
        #保存标志
        self.is_save = True 
        #读入文件标志
        self.is_read = False
        #数据集是否已经分割标志
        self.is_split_data = False
        #数据集分割索引
        self.data_label_cur_number = None
        #半监督
        self.is_semi_supervised_learning_data_split = False
        self.cur_semi_supervised_learning_field_index = None
        self.cur_semi_supervised_learning_fieldvalue_index = None
        self.is_zero_field2 = False

    #############################################菜单操作#######################################################
    #读取文件
    @pyqtSlot()
    def on_action_csvfile_triggered(self):
        filename, type = QFileDialog.getOpenFileName(self, "选择文件", "/home", ".CSV文件(*.csv)")
        #保存标志
        self.is_save = False
        #读入文件标志
        self.is_read = True
        #filename = "./HR2.csv"
        #filename = "./Fremont.csv"
        if filename != "":
            # 查看模型
            #self.df = pd.read_excel(filename)
            self.df = pd.read_csv(filename)
            self.copydf = self.df.copy()
            self.dataframemodel.setDataFrame(self.copydf)
            self.ui_interfacewin.widget_datatablemodel.setButtonsVisible(True)
            # 分析模型
            self.standardmodel.clear()

            rownum, colnum = self.copydf.shape
            col_names = self.copydf.columns.tolist()
            if rownum < 1000 or rownum == 1000:
                for col in range(colnum):
                    colname = []
                    rows = self.copydf[col_names[col]].tolist()
                    for row in range(rownum):
                        item = QStandardItem("%s" % (rows[row]))
                        colname.append(item)
                    self.standardmodel.insertColumn(col, colname)
            if rownum > 1000:
                for col in range(colnum):
                    colname = []
                    rows = self.copydf[col_names[col]].tolist()
                    for row in range(1, 1000):
                        item = QStandardItem("%s" % (rows[row]))
                        colname.append(item)
                    self.standardmodel.insertColumn(col, colname)
            self.standardmodel.setHorizontalHeaderLabels(col_names)
            #俩组数组检验
            self.ui_interfacewin.comboBox_test_col.clear()
            self.ui_interfacewin.comboBox_test_col.addItems(col_names)
            #正态性检验
            self.ui_interfacewin.listWidget_normality_test.clear()
            self.ui_interfacewin.listWidget_normality_test_2.clear()
            self.ui_interfacewin.listWidget_normality_test.addItems(col_names)
            #方差检验
            self.ui_interfacewin.listWidget_std_test.clear()
            self.ui_interfacewin.listWidget_std_test_2.clear()
            self.ui_interfacewin.listWidget_std_test.addItems(col_names)
            #多组数组检验
            self.ui_interfacewin.listWidget_many_numbers_compare.clear()
            self.ui_interfacewin.listWidget_many_numbers_compare_2.clear()
            self.ui_interfacewin.listWidget_many_numbers_compare.addItems(col_names)
            #相关性检验
            self.ui_interfacewin.listWidget_corr_test_2.clear()
            self.ui_interfacewin.listWidget_corr_test.clear()
            self.ui_interfacewin.listWidget_corr_test.addItems(col_names)
            #二元值与连续值之间的关系
            self.ui_interfacewin.listWidget_Binary_and_continuous_values_test.clear()
            self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.clear()
            self.ui_interfacewin.listWidget_Binary_and_continuous_values_test.addItems(col_names)
            #异常值处理
            self.ui_interfacewin.listWidget_detection_outlier.clear()
            self.ui_interfacewin.listWidget_detection_outlier_2.clear()
            self.ui_interfacewin.listWidget_detection_outlier.addItems(col_names)
            #空值处理
            self.ui_interfacewin.listWidget_null_values_2.clear()
            self.ui_interfacewin.listWidget_null_values.clear()
            self.ui_interfacewin.listWidget_null_values.addItems(col_names)
            #线性回归
            self.ui_interfacewin.listWidget_linear_regression_2.clear()
            self.ui_interfacewin.listWidget_linear_regression.clear()
            self.ui_interfacewin.listWidget_linear_regression.addItems(col_names)
            #PCA降维
            self.ui_interfacewin.listWidget_PCA.clear()
            self.ui_interfacewin.listWidget_PCA_2.clear()
            self.ui_interfacewin.listWidget_PCA.addItems(col_names)
            #取样
            self.ui_interfacewin.listWidget_sample.clear()
            self.ui_interfacewin.listWidget_sample_2.clear()
            self.ui_interfacewin.listWidget_sample.addItems(col_names)
            #离散特征相关性分析
            self.ui_interfacewin.listWidget_discrete_correlation.clear()
            self.ui_interfacewin.listWidget_discrete_correlation_2.clear()
            self.ui_interfacewin.listWidget_discrete_correlation.addItems(col_names)
            #LDA降维
            self.ui_interfacewin.listWidget_LDA.clear()
            self.ui_interfacewin.listWidget_LDA_2.clear()
            self.ui_interfacewin.listWidget_LDA.addItems(col_names)
            #特征选择
            self.ui_interfacewin.listWidget_features_selection.clear()
            self.ui_interfacewin.listWidget_features_selection_2.clear()
            self.ui_interfacewin.listWidget_features_selection_3.clear()
            self.ui_interfacewin.listWidget_features_selection.addItems(col_names)
            #特征变换
            self.ui_interfacewin.listWidget_features_transform.clear()
            self.ui_interfacewin.listWidget_features_transform_2.clear()
            self.ui_interfacewin.listWidget_features_transform.addItems(col_names)
            #建立模型选择标注
            self.is_model_label_null = False
            self.ui_interfacewin.comboBox_model_label.clear()
            self.ui_interfacewin.comboBox_model_label.addItems(col_names)
            #关联分析
            self.ui_interfacewin.listWidget_correlation_analysis_Apriori.clear()
            self.ui_interfacewin.listWidget_correlation_analysis_Apriori_2.clear()
            self.ui_interfacewin.listWidget_correlation_analysis_Apriori.addItems(col_names)

    #显示操作窗口
    @pyqtSlot()
    def on_action_showopeatorwin_triggered(self):
        if self.ui_interfacewin.dockWidget.isHidden():
            self.ui_interfacewin.dockWidget.show()

    @pyqtSlot()
    def on_action_save_triggered(self):
        if self.is_read:
            if self.is_save:
                self.copydf.to_csv(index=False)
                self.ui_interfacewin.textEdit_result.append("保存成功")
            else:
                savefile = SaveFile()
                savefile.open()
                if savefile.exec() == QDialog.Accepted:
                    sep, na_rep, encoding, compression, decimal, savefilename, index, header = savefile.save()
                    self.copydf.to_csv(path_or_buf=savefilename, sep=sep, na_rep=na_rep, header=header, index=index,
                                        encoding=encoding, compression=compression, decimal=decimal)
                    self.ui_interfacewin.textEdit_result.append("保存成功：%s" % (savefilename))
                    self.is_save = True
    @pyqtSlot()
    def on_action_saveas_triggered(self):
        if self.is_read:
            savefile = SaveFile()
            savefile.open()
            if savefile.exec() == QDialog.Accepted:
                sep, na_rep, encoding, compression, decimal, savefilename, index, header = savefile.save()
                self.copydf.to_csv(path_or_buf=savefilename, sep=sep, na_rep=na_rep, header=header, index=index,
                                   encoding=encoding, compression=compression, decimal=decimal)
                self.ui_interfacewin.textEdit_result.append("另存为成功：%s" % (savefilename))
    #############################################菜单操作#######################################################
    # 刷新
    def refresh(self):
        """
        # 查看模型
        self.df = pd.read_csv(filename)
        self.copydf = self.df.copy()
        self.dataframemodel.setDataFrame(self.copydf)
        self.ui_interfacewin.widget_datatablemodel.setButtonsVisible(True)
        """
        # 分析模型
        self.standardmodel.clear()
        rownum, colnum = self.copydf.shape
        col_names = self.copydf.columns.tolist()
        if rownum < 1000 or rownum == 1000:
            for col in range(colnum):
                colname = []
                rows = self.copydf[col_names[col]].tolist()
                for row in range(rownum):
                    item = QStandardItem("%s" % (rows[row]))
                    colname.append(item)
                self.standardmodel.insertColumn(col, colname)
        if rownum > 1000:
            for col in range(colnum):
                colname = []
                rows = self.copydf[col_names[col]].tolist()
                for row in range(1, 1000):
                    item = QStandardItem("%s" % (rows[row]))
                    colname.append(item)
                self.standardmodel.insertColumn(col, colname)
        self.standardmodel.setHorizontalHeaderLabels(col_names)
        # 俩组数组检验
        self.ui_interfacewin.comboBox_test_col.clear()
        self.ui_interfacewin.comboBox_test_col.addItems(col_names)
        # 正态性检验
        self.ui_interfacewin.listWidget_normality_test.clear()
        self.ui_interfacewin.listWidget_normality_test_2.clear()
        self.ui_interfacewin.listWidget_normality_test.addItems(col_names)
        # 方差检验
        self.ui_interfacewin.listWidget_std_test.clear()
        self.ui_interfacewin.listWidget_std_test_2.clear()
        self.ui_interfacewin.listWidget_std_test.addItems(col_names)
        # 多组数组检验
        self.ui_interfacewin.listWidget_many_numbers_compare.clear()
        self.ui_interfacewin.listWidget_many_numbers_compare_2.clear()
        self.ui_interfacewin.listWidget_many_numbers_compare.addItems(col_names)
        # 相关性检验
        self.ui_interfacewin.listWidget_corr_test_2.clear()
        self.ui_interfacewin.listWidget_corr_test.clear()
        self.ui_interfacewin.listWidget_corr_test.addItems(col_names)
        # 二元值与连续值之间的关系
        self.ui_interfacewin.listWidget_Binary_and_continuous_values_test.clear()
        self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.clear()
        self.ui_interfacewin.listWidget_Binary_and_continuous_values_test.addItems(col_names)
        # 异常值处理
        self.ui_interfacewin.listWidget_detection_outlier.clear()
        self.ui_interfacewin.listWidget_detection_outlier_2.clear()
        self.ui_interfacewin.listWidget_detection_outlier.addItems(col_names)
        # 空值处理
        self.ui_interfacewin.listWidget_null_values_2.clear()
        self.ui_interfacewin.listWidget_null_values.clear()
        self.ui_interfacewin.listWidget_null_values.addItems(col_names)
            
    #操作工具栏变换
    @pyqtSlot()
    def on_toolBox_currentChanged(self):
        if self.ui_interfacewin.toolBox.currentIndex()==0:
            self.ui_interfacewin.stackedWidget.setCurrentIndex(0)

    # 单因子数据分析单显示
    @pyqtSlot()
    def on_signal_show_Btn_clicked(self):
        ismean = self.ui_interfacewin.checkBox_mean.isChecked()
        ismedian = self.ui_interfacewin.checkBox_median.isChecked()
        isquantile = self.ui_interfacewin.checkBox_quantile.isChecked()
        isquantile1 = self.ui_interfacewin.checkBox_quantile1.isChecked()
        ismode = self.ui_interfacewin.checkBox_mode.isChecked()
        isstd = self.ui_interfacewin.checkBox_std.isChecked()
        isvar = self.ui_interfacewin.checkBox_var.isChecked()
        isskew = self.ui_interfacewin.checkBox_skew.isChecked()
        iskurt = self.ui_interfacewin.checkBox_kurt.isChecked()
        issum = self.ui_interfacewin.checkBox_sum.isChecked()

        itemcount = self.ui_interfacewin.listWidget_opeator_col.count()
        for i in range(0, itemcount):
            itemtext = self.ui_interfacewin.listWidget_opeator_col.item(i).text()
            self.ui_interfacewin.textEdit_result.append("%s:\n" % (itemtext))
            if ismean:
                mean = self.copydf[itemtext].mean()
                self.ui_interfacewin.textEdit_result.append("平均数:%f" % (mean))
            if ismedian:
                median = self.copydf[itemtext].median()
                self.ui_interfacewin.textEdit_result.append("中位数:%f" % (median))
            if ismode:
                mode = self.copydf[itemtext].mode()
                self.ui_interfacewin.textEdit_result.append("众数:%f" % (mode))
            if isquantile:
                quantile = self.copydf[itemtext].quantile(0.25)
                self.ui_interfacewin.textEdit_result.append("上四分位数:%f" % (quantile))
            if isquantile1:
                quantile1 = self.copydf[itemtext].mode()
                self.ui_interfacewin.textEdit_result.append("下四分位数%f" % (quantile1))
            if issum:
                sum = self.copydf[itemtext].sum()
                self.ui_interfacewin.textEdit_result.append("和:%f" % (sum))
            if isstd:
                std = self.copydf[itemtext].std()
                self.ui_interfacewin.textEdit_result.append("标准差:%f" % (std))
            if isvar:
                var = self.copydf[itemtext].var()
                self.ui_interfacewin.textEdit_result.append("方差:%f" % (var))
            if isskew:
                skew = self.copydf[itemtext].skew()
                self.ui_interfacewin.textEdit_result.append("偏态:%f" % (skew))
            if iskurt:
                kurt = self.copydf[itemtext].kurt()
                self.ui_interfacewin.textEdit_result.append("峰态:%f\n" % (kurt))
        # 获取选择条目的列标题
        """
        index = self.ui_interfacewin.tableView_standardmodel.selectionModel().selectedIndexes()
        print(index)
        for i in range(len(index)):
            c = index[i].column()
            g = self.standardmodel.horizontalHeaderItem(c)
            print(g.text())
            """
    # 单因子数据分析多显示
    @pyqtSlot()
    def on_Between_show_Btn_clicked(self):
        ismean = self.ui_interfacewin.checkBox_mean.isChecked()
        ismedian = self.ui_interfacewin.checkBox_median.isChecked()
        isquantile = self.ui_interfacewin.checkBox_quantile.isChecked()
        isquantile1 = self.ui_interfacewin.checkBox_quantile1.isChecked()
        ismode = self.ui_interfacewin.checkBox_mode.isChecked()
        isstd = self.ui_interfacewin.checkBox_std.isChecked()
        isvar = self.ui_interfacewin.checkBox_var.isChecked()
        isskew = self.ui_interfacewin.checkBox_skew.isChecked()
        iskurt = self.ui_interfacewin.checkBox_kurt.isChecked()
        issum = self.ui_interfacewin.checkBox_sum.isChecked()

        itemcount = self.ui_interfacewin.listWidget_opeator_col.count()
        listcol = []
        for i in range(0, itemcount):
            itemtext = self.ui_interfacewin.listWidget_opeator_col.item(i).text()
            str = itemtext
            listcol.append(str)
        copydf1 = self.copydf[listcol]
        if ismean:
            mean = copydf1.mean()
            self.ui_interfacewin.textEdit_result.append("平均数:%f" % (mean))
        if ismedian:
            median = copydf1.median()
            self.ui_interfacewin.textEdit_result.append("中位数:%f" % (median))
        if ismode:
            mode = copydf1.mode()
            self.ui_interfacewin.textEdit_result.append("众数:%f" % (mode))
        if isquantile:
            quantile = copydf1.quantile(0.25)
            self.ui_interfacewin.textEdit_result.append("上四分位数:%f"%(quantile))
        if isquantile1:
            quantile1 = copydf1.quantile(0.75)
            self.ui_interfacewin.textEdit_result.append("下四分位数:%f"%(quantile1))
        if issum:
            sum = copydf1.sum()
            self.ui_interfacewin.textEdit_result.append("和:%f" % (sum))
        if isstd:
            std = copydf1.std()
            self.ui_interfacewin.textEdit_result.append("标准差:%f" % (std))
        if isvar:
            var = copydf1.var()
            self.ui_interfacewin.textEdit_result.append("方差:%f" % (var))
        if isskew:
            skew = copydf1.skew()
            self.ui_interfacewin.textEdit_result.append("偏态:%f" % (skew))
        if iskurt:
            kurt = copydf1.kurt()
            self.ui_interfacewin.textEdit_result.append("峰态:%f\n" % (kurt))

    # 单因子数据分析全部显示
    @pyqtSlot()
    def on_all_show_Btn_clicked(self):
        #print("全部数据\n")
        mean = self.copydf.mean()
        self.ui_interfacewin.textEdit_result.append("平均数:%s" % (str(mean)))
        median = self.copydf.median()
        self.ui_interfacewin.textEdit_result.append("中位数:%s" % (str(median)))
        mode = self.copydf.mode()
        self.ui_interfacewin.textEdit_result.append("众数:%s" % (str(mode)))
        quantitle = self.copydf.quantile(0.25)
        self.ui_interfacewin.textEdit_result.append("上四分位数:%s" % (str(quantitle)))
        quantitle1 = self.copydf.quantile(0.75)
        self.ui_interfacewin.textEdit_result.append("下四分位数:%s" % (str(quantitle1)))
        sum = self.copydf.sum()
        self.ui_interfacewin.textEdit_result.append("和:%s" % (str(sum)))
        std = self.copydf.std()
        self.ui_interfacewin.textEdit_result.append("标准差:%s" % (str(std)))
        var = self.copydf.var()
        self.ui_interfacewin.textEdit_result.append("方差:%s" % (str(var)))
        skew = self.copydf.skew()
        self.ui_interfacewin.textEdit_result.append("偏态:%s" % (str(skew)))
        kurt = self.copydf.kurt()
        self.ui_interfacewin.textEdit_result.append("峰态:%s\n" % (str(kurt)))

    #单因子数据分析添加操作列
    @pyqtSlot(QModelIndex)
    def on_tableView_standardmodel_clicked(self, index):
        if self.ui_interfacewin.checkBox.isChecked():
            col = index.column()
            colheader = self.standardmodel.horizontalHeaderItem(col).text()
            self.ui_interfacewin.listWidget_opeator_col.addItem(colheader)
    #删除操作项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_opeator_col_itemClicked(self, itemlist):
        row = self.ui_interfacewin.listWidget_opeator_col.row(itemlist)
        self.ui_interfacewin.listWidget_opeator_col.takeItem(row)
    #单因子与多因子分析切换
    @pyqtSlot(int)
    def on_comboBox_currentIndexChanged(self, index):
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(index)
            if index == 1:
                self.ui_interfacewin.comboBox_querymode.hide()
                self.ui_interfacewin.listWidget_opeator_col.hide()
                self.ui_interfacewin.checkBox.hide()
            if index == 0:
                self.ui_interfacewin.comboBox_querymode.show()
                self.ui_interfacewin.listWidget_opeator_col.show()
                self.ui_interfacewin.checkBox.show()

    #多因子数据分析模块************************************************************************************************
    #多因子数据分析类型切换
    @pyqtSlot(int)
    def on_comboBox_Multiple_factor_analysis_type_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(2)
        if index ==1:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(3)
        if index == 2:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(1)
        if index == 3:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(4)
        if index == 4:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(5)
        if index == 5:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(6)
        if index == 6:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(9)
        if index == 7:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(10)
        if index == 8:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(11)
        if index == 9:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(12)
    #俩组数组比较#开始
    #添加独立检验col
    @pyqtSlot()
    def on_add_test_col_Btn_clicked(self):
        colname = self.ui_interfacewin.comboBox_test_col.currentText()
        chioce = self.ui_interfacewin.checkBox_add_test_col.isChecked()
        if chioce:
            self.ui_interfacewin.listWidget_add_test_field.addItem(colname)
        else:
            self.ui_interfacewin.listWidget_add_test_col.addItem(colname)
    #删除独立检验col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_add_test_col_itemDoubleClicked(self, item):
        currow = self.ui_interfacewin.listWidget_add_test_col.currentIndex().row()
        self.ui_interfacewin.listWidget_add_test_col.takeItem(currow)

    #删除独立检验col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_add_test_field_itemClicked(self, item):
        currow = self.ui_interfacewin.listWidget_add_test_field.currentIndex().row()
        self.ui_interfacewin.listWidget_add_test_field.takeItem(currow)
    #添加独立分布检验字段
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_add_test_col_itemClicked(self, item):
        self.ui_interfacewin.listWidget_add_test_fields.clear()
        self.ui_interfacewin.listWidget_add_test_field1.clear()
        colname = item.text()
        str1 = str(colname)
        field = self.copydf[str1].unique()

        str2 = field.tolist()
        new_str2 = [str(x) for x in str2]
        self.ui_interfacewin.listWidget_add_test_fields.addItems(new_str2)

    #选择独立分布检验字段
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_add_test_fields_itemClicked(self, item):
        colname = item.text()
        self.ui_interfacewin.listWidget_add_test_field1.addItem(colname)

    # 删除独立检验字段
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_add_test_field1_itemClicked(self, index):
        currow = self.ui_interfacewin.listWidget_add_test_field1.currentIndex().row()
        self.ui_interfacewin.listWidget_add_test_field1.takeItem(currow)

    #独立俩样本t检验#结束
    #俩组数组比较条件切换
    @pyqtSlot(int)
    def on_comboBox_two_numbers_compare_currentIndexChanged(self, index):
        curcomparetype = self.ui_interfacewin.comboBox_two_numbers_compare.currentIndex()
        if curcomparetype == 0:
            self.ui_interfacewin.checkBox_equal_var.setEnabled(True)
            self.ui_interfacewin.checkBox_equal_var.setText("equal_var")
        if curcomparetype == 1:
            self.ui_interfacewin.checkBox_equal_var.setEnabled(False)
        if curcomparetype == 2:
            self.ui_interfacewin.checkBox_equal_var.setEnabled(True)
            self.ui_interfacewin.checkBox_equal_var.setText("equal_var")
        if curcomparetype == 3:
            self.ui_interfacewin.checkBox_equal_var.setEnabled(False)
        if curcomparetype == 4:
            self.ui_interfacewin.checkBox_equal_var.setEnabled(True)
            self.ui_interfacewin.checkBox_equal_var.setText("use_continuity")
    #全部显示
    @pyqtSlot()
    def on_all_test_btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_add_test_col.count()
        field1count = self.ui_interfacewin.listWidget_add_test_field.count()
        curcomparetype = self.ui_interfacewin.comboBox_two_numbers_compare.currentIndex()
        # 独立俩样本t检验
        if curcomparetype==0:
            for i in range(0, fieldcount):
                for j in range(0, field1count):
                    if i==j:
                        field = self.ui_interfacewin.listWidget_add_test_col.item(i).text()
                        field1 = self.ui_interfacewin.listWidget_add_test_field.item(j).text()
                        fieldvalues = self.copydf[field]
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype = fieldvalues.dtype.name
                        fieldvalues1type = fieldvalues1.dtype.name
                        equal_var = self.ui_interfacewin.checkBox_equal_var.isChecked()
                        if fieldvaluestype != 'object':
                            if fieldvalues1type != 'object':
                                #rvs1 = ss.norm.rvs(fieldvalues)
                                #rvs2 = ss.norm.rvs(fieldvalues1)
                                #print(equal_var)
                                ttt = ss.ttest_ind(fieldvalues, fieldvalues1,equal_var=equal_var)
                                self.ui_interfacewin.textEdit_result.append("%s列和%s列的独立俩样本t检验:"%(field,field1))
                                self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(ttt))
                            else:
                                QMessageBox.critical(self, "错误","%s字段内容中含有字符串!"%(field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, "错误", "%s字段内容中含有字符串!"%(field), QMessageBox.Ok)
        #成对俩样本t检验
        if curcomparetype==1:
            for i in range(0, fieldcount):
                for j in range(0, field1count):
                    if i == j:
                        field = self.ui_interfacewin.listWidget_add_test_col.item(i).text()
                        field1 = self.ui_interfacewin.listWidget_add_test_field.item(j).text()
                        fieldvalues = self.copydf[field]
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype = fieldvalues.dtype.name
                        fieldvalues1type = fieldvalues1.dtype.name
                        if fieldvaluestype != 'object':
                            if fieldvalues1type != 'object':
                                # rvs1 = ss.norm.rvs(fieldvalues)
                                # rvs2 = ss.norm.rvs(fieldvalues1)
                                # print(equal_var)
                                ttt = ss.ttest_rel(fieldvalues, fieldvalues1)
                                self.ui_interfacewin.textEdit_result.append("%s列和%s列的成对俩样本t检验:" % (field, field1))
                                self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(ttt))
                            else:
                                QMessageBox.critical(self, "错误", "%s字段内容中含有字符串!" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, "错误", "%s字段内容中含有字符串!" % (field), QMessageBox.Ok)
        #通过基本统计量来做独立俩样本检验
        if curcomparetype == 2:
            for i in range(0, fieldcount):
                for j in range(0, field1count):
                    if i == j:
                        field = self.ui_interfacewin.listWidget_add_test_col.item(i).text()
                        field1 = self.ui_interfacewin.listWidget_add_test_field.item(j).text()
                        fieldvalues = self.copydf[field]
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype = fieldvalues.dtype.name
                        fieldvalues1type = fieldvalues1.dtype.name
                        equal_var = self.ui_interfacewin.checkBox_equal_var.isChecked()
                        if fieldvaluestype != 'object':
                            if fieldvalues1type != 'object':
                                # rvs1 = ss.norm.rvs(fieldvalues)
                                # rvs2 = ss.norm.rvs(fieldvalues1)
                                # print(equal_var)
                                fieldmean = fieldvalues.mean()
                                fieldmean1 = fieldvalues1.mean()
                                fieldstd = fieldvalues.std()
                                fieldstd1 = fieldvalues1.std()


                                row = len(fieldvalues)
                                row1 = len(fieldvalues1)

                                ttt = ss.ttest_ind_from_stats(mean1=fieldmean, std1=fieldstd, nobs1=row,
                                                              mean2=fieldmean1, std2=fieldstd1, nobs2=row1, equal_var=equal_var)
                                self.ui_interfacewin.textEdit_result.append("%s列和%s列的通过基本统计量来做独立俩样本检验:" % (field, field1))
                                self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(ttt))
                            else:
                                QMessageBox.critical(self, "错误", "%s字段内容中含有字符串!" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, "错误", "%s字段内容中含有字符串!" % (field), QMessageBox.Ok)
        if curcomparetype == 3:
            for i in range(0, fieldcount):
                for j in range(0, field1count):
                    if i==j:
                        field = self.ui_interfacewin.listWidget_add_test_col.item(i).text()
                        field1 = self.ui_interfacewin.listWidget_add_test_field.item(j).text()
                        fieldvalues = self.copydf[field]
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype = fieldvalues.dtype.name
                        fieldvalues1type = fieldvalues1.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvalues1type != 'object':
                                ttt = ss.ranksums(fieldvalues,fieldvalues1)
                                self.ui_interfacewin.textEdit_result.append("%s列和%s列的wilcox秩序和检验:"%(field,field1))
                                self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(ttt))
                            else:
                                QMessageBox.critical(self, "错误","%s字段内容中含有字符串!"%(field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, "错误", "%s字段内容中含有字符串!"%(field), QMessageBox.Ok)
        if curcomparetype == 4:
            for i in range(0, fieldcount):
                for j in range(0, field1count):
                    if i==j:
                        field = self.ui_interfacewin.listWidget_add_test_col.item(i).text()
                        field1 = self.ui_interfacewin.listWidget_add_test_field.item(j).text()
                        fieldvalues = self.copydf[field]
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype = fieldvalues.dtype.name
                        fieldvalues1type = fieldvalues1.dtype.name
                        use_continuity = self.ui_interfacewin.checkBox_equal_var.isChecked()
                        if fieldvaluestype != 'object':
                            if fieldvalues1type != 'object':
                                ttt = ss.mstats.mannwhitneyu(fieldvalues, fieldvalues1, use_continuity=use_continuity)
                                self.ui_interfacewin.textEdit_result.append("%s列和%s列的Mann-Whitney U检测:"%(field,field1))
                                self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(ttt))
                            else:
                                QMessageBox.critical(self, "错误","%s字段内容中含有字符串!"%(field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, "错误", "%s字段内容中含有字符串!"%(field), QMessageBox.Ok)

    #独立分布检验显示
    @pyqtSlot()
    def on_test_Btn_clicked(self):
        curcomparetype = self.ui_interfacewin.comboBox_two_numbers_compare.currentIndex()
        #独立俩样本t检验
        if curcomparetype == 0:
            equal_var = self.ui_interfacewin.checkBox_equal_var.isChecked()
            col = self.ui_interfacewin.listWidget_add_test_col.currentItem().text()
            indices = self.copydf.groupby(by=col).indices

            fieldcount = self.ui_interfacewin.listWidget_add_test_field.count()
            fieldcount2 = self.ui_interfacewin.listWidget_add_test_field1.count()
            fieldcount3 = fieldcount2-1
            for i in range(0, fieldcount):

                fieldtext = self.ui_interfacewin.listWidget_add_test_field.item(i).text()
                for j in range(0, fieldcount3):
                    val = j+1
                    field1text = self.ui_interfacewin.listWidget_add_test_field1.item(j).text()
                    fieldtext1 = self.ui_interfacewin.listWidget_add_test_field1.item(val).text()

                    if is_number(fieldtext1):
                        aaa = float(fieldtext1)
                        fieldvalues = self.copydf[fieldtext].iloc[indices[aaa]].values
                    else:
                        field1values = self.copydf[fieldtext].iloc[indices[fieldtext1]].values

                    if is_number(field1text):
                        bbb = float(field1text)
                        field1values = self.copydf[fieldtext].iloc[indices[bbb]].values
                    else:
                        fieldvalues = self.copydf[fieldtext].iloc[indices[field1text]].values

                    self.ui_interfacewin.textEdit_result.append("%s列的(%s和%s)对应于%s列的独立俩样本T检验："%(col, field1text, fieldtext1,fieldtext))
                    self.ui_interfacewin.textEdit_result.append(str(ss.ttest_ind(fieldvalues,field1values,equal_var=equal_var)))
                    """                   
                    keys = list(indices.keys())
                    lenkeys = len(keys)
                    mat = np.zeros((lenkeys, lenkeys))
                    for h in range(lenkeys):
                        for k in range(lenkeys):
                            p_value = ss.ttest_ind(self.copydf[fieldtext].iloc[indices[keys[h]]].values,self.copydf[fieldtext].iloc[indices[keys[k]]].values,equal_var=equal_var)[1]
                            if p_value<0.05:
                                mat[h][k] = -1
                            else:
                                mat[h][k] = p_value
                    sns.heatmap(mat,xticklabels=keys,yticklabels=keys)
                    plt.show()
                    """
    #俩组数组比较#结束
    #正态性检验
    #添加正态检验col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_normality_test_itemClicked(self, item):
        itemtext = item.text()
        self.ui_interfacewin.listWidget_normality_test_2.addItem(itemtext)
    #删除正态检验列
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_normality_test_2_itemClicked(self, item):
        index = self.ui_interfacewin.listWidget_normality_test_2.currentIndex().row()
        self.ui_interfacewin.listWidget_normality_test_2.takeItem(index)
    #正态检验筛选框变化
    @pyqtSlot(int)
    def on_comboBox_normality_test_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.groupBox_5.show()
            self.ui_interfacewin.groupBox_6.show()
        if index == 1:
            self.ui_interfacewin.groupBox_5.hide()
            self.ui_interfacewin.groupBox_6.hide()
        if index == 2:
            self.ui_interfacewin.groupBox_5.hide()
            self.ui_interfacewin.groupBox_6.hide()
        if index == 3:
            self.ui_interfacewin.groupBox_5.hide()
            self.ui_interfacewin.groupBox_6.hide()

    #正态性检验
    @pyqtSlot()
    def on_normality_test_Btn_clicked(self):
        normtype = self.ui_interfacewin.comboBox_normality_test.currentIndex()
        itemcount = self.ui_interfacewin.listWidget_normality_test_2.count()
        #K-S检验
        if normtype == 0:
            #alternative参数
            if self.ui_interfacewin.radioButton_KS_two_sided.isChecked():
                alternative = "two-sided"
            if self.ui_interfacewin.radioButton_KS_less.isChecked():
                alternative = "less"
            if self.ui_interfacewin.radioButton_KS_greater.isChecked():
                alternative = "greater"
            #mode参数
            if self.ui_interfacewin.radioButton_KS_approx.isChecked():
                mode = "approx"
            if self.ui_interfacewin.radioButton_KS_asymp.isChecked():
                mode = "asymp"

            for i in range(0, itemcount):
                colname = self.ui_interfacewin.listWidget_normality_test_2.item(i).text()
                colvalues = self.copydf[colname]
                colvaluestype = colvalues.dtype.name
                if colvaluestype != 'object':
                    hhh = ss.kstest(colvalues, 'norm',alternative=alternative, mode=mode)
                    self.ui_interfacewin.textEdit_result.append("%s列的K-S检验:"%(colname))
                    self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(hhh))
        #Shapiro
        if normtype == 1:
            for i in range(0, itemcount):
                colname = self.ui_interfacewin.listWidget_normality_test_2.item(i).text()
                colvalues = self.copydf[colname]
                colvaluestype = colvalues.dtype.name
                if colvaluestype != 'object':
                    hhh = ss.shapiro(colvalues)
                    self.ui_interfacewin.textEdit_result.append("%s列的Shapiro检验:"%(colname))
                    self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(hhh))
        if normtype == 2:
            for i in range(0, itemcount):
                colname = self.ui_interfacewin.listWidget_normality_test_2.item(i).text()
                colvalues = self.copydf[colname]
                colvaluestype = colvalues.dtype.name
                if colvaluestype != 'object':
                    hhh = ss.normaltest(colvalues)
                    self.ui_interfacewin.textEdit_result.append("%s列的Normal检验:"%(colname))
                    self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(hhh))
        if normtype == 3:
            for i in range(0, itemcount):
                colname = self.ui_interfacewin.listWidget_normality_test_2.item(i).text()
                colvalues = self.copydf[colname]
                colvaluestype = colvalues.dtype.name
                if colvaluestype != 'object':
                    hhh = ss.anderson(colvalues, dist='norm')
                    self.ui_interfacewin.textEdit_result.append("%s列的Anderson检验:"%(colname))
                    self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(hhh))
    #方差检验
    #添加列
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_std_test_itemClicked(self, item):
        itemname = item.text()
        self.ui_interfacewin.listWidget_std_test_2.addItem(itemname)
    #删除列
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_std_test_2_itemClicked(self, item):
        index = self.ui_interfacewin.listWidget_std_test_2.currentIndex().row()
        self.ui_interfacewin.listWidget_std_test_2.takeItem(index)
    #方差齐次检验条件框变化
    @pyqtSlot(int)
    def on_comboBox_std_test_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.groupBox_7.hide()
        if index == 1:
            self.ui_interfacewin.groupBox_7.show()
        if index == 2:
            self.ui_interfacewin.groupBox_7.show()
    #检验
    @pyqtSlot()
    def on_std_test_Btn_clicked(self):
        curtypeindex = self.ui_interfacewin.comboBox_std_test.currentIndex()
        if curtypeindex == 0:
            stditemcount = self.ui_interfacewin.listWidget_std_test_2.count()
            stditemcount2 = stditemcount-1
            for i in range(0, stditemcount2):
                colname = self.ui_interfacewin.listWidget_std_test_2.item(i).text()
                j = i+1
                colname2 = self.ui_interfacewin.listWidget_std_test_2.item(j).text()
                colvalues = self.copydf[colname]
                colvalues2 = self.copydf[colname]
                colvaluestype2 = colvalues2.dtype.name
                colvaluestype = colvalues.dtype.name
                if colvaluestype != 'object':
                    if colvaluestype2 != 'object':
                        hhh = ss.bartlett(colvalues,colvalues2)
                        self.ui_interfacewin.textEdit_result.append("%s列和%s列的Bartlett检验:" % (colname, colname2))
                        self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(hhh))
                    else:
                        QMessageBox.critical(self, 'Error', "%s列有字符"%(colname2), QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, 'Error', "%s列有字符"%(colname), QMessageBox.Ok)
        #Levene检验
        if curtypeindex == 1:
            if self.ui_interfacewin.radioButton_levene_mean.isChecked():
                center = 'mean'
            if self.ui_interfacewin.radioButton_levene_median.isChecked():
                center = 'median'
            if self.ui_interfacewin.radioButton_levene_trimmed.isChecked():
                center = 'trimmed'
            stditemcount = self.ui_interfacewin.listWidget_std_test_2.count()
            stditemcount2 = stditemcount-1
            for i in range(0, stditemcount2):
                colname = self.ui_interfacewin.listWidget_std_test_2.item(i).text()
                j = i + 1
                colname2 = self.ui_interfacewin.listWidget_std_test_2.item(j).text()
                colvalues = self.copydf[colname]
                colvalues2 = self.copydf[colname]
                colvaluestype2 = colvalues2.dtype.name
                colvaluestype = colvalues.dtype.name
                if colvaluestype != 'object':
                    if colvaluestype2 != 'object':
                        hhh = ss.levene(colvalues, colvalues2, center=center)
                        self.ui_interfacewin.textEdit_result.append("%s列和%s列的Levene检验:" % (colname, colname2))
                        self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(hhh))
                    else:
                        QMessageBox.critical(self, 'Error', "%s列有字符" % (colname2), QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, 'Error', "%s列有字符" % (colname), QMessageBox.Ok)
        # Fligner-Killeen检验
        if curtypeindex == 2:
            if self.ui_interfacewin.radioButton_levene_mean.isChecked():
                center = 'mean'
            if self.ui_interfacewin.radioButton_levene_median.isChecked():
                center = 'median'
            if self.ui_interfacewin.radioButton_levene_trimmed.isChecked():
                center = 'trimmed'
            stditemcount = self.ui_interfacewin.listWidget_std_test_2.count()
            stditemcount2 = stditemcount - 1
            for i in range(0, stditemcount2):
                colname = self.ui_interfacewin.listWidget_std_test_2.item(i).text()
                j = i + 1
                colname2 = self.ui_interfacewin.listWidget_std_test_2.item(j).text()
                colvalues = self.copydf[colname]
                colvalues2 = self.copydf[colname]
                colvaluestype2 = colvalues2.dtype.name
                colvaluestype = colvalues.dtype.name
                if colvaluestype != 'object':
                    if colvaluestype2 != 'object':
                        hhh = ss.fligner(colvalues, colvalues2, center=center)
                        self.ui_interfacewin.textEdit_result.append("%s列和%s列的Fligner-Killeen检验:" % (colname, colname2))
                        self.ui_interfacewin.textEdit_result.append('statistic=%f, pvalue=%f'%(hhh))
                    else:
                        QMessageBox.critical(self, 'Error', "%s列有字符" % (colname2), QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, 'Error', "%s列有字符" % (colname), QMessageBox.Ok)

    #多组数组比较
    #添加col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_many_numbers_compare_itemClicked(self, item):
        itemname = item.text()
        self.ui_interfacewin.listWidget_many_numbers_compare_2.addItem(itemname)
    #删除col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_many_numbers_compare_2_itemClicked(self, item):
        row = self.ui_interfacewin.listWidget_many_numbers_compare_2.currentIndex().row()
        self.ui_interfacewin.listWidget_many_numbers_compare_2.takeItem(row)

    #检验
    @pyqtSlot()
    def on_many_numbers_compare_Btn_clicked(self):
        curtypeindex = self.ui_interfacewin.comboBox_many_numbers_compare.currentIndex()
        itemcount = self.ui_interfacewin.listWidget_many_numbers_compare_2.count()
        if curtypeindex == 0:
            #2组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_2.isChecked():
                if itemcount < 2:
                    QMessageBox.critical(self, 'Error', "至少要选择2列", QMessageBox.Ok)
                else:
                    for i in range(1, 2):
                        j= i-1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                result = ss.f_oneway(fieldvalues, fieldvalues1)
                                self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                self.ui_interfacewin.textEdit_result.append(str(result))
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            #3组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_3.isChecked():
                if itemcount < 3:
                    QMessageBox.critical(self, 'Error', "至少要选择3列", QMessageBox.Ok)
                else:
                    for i in range(2, 3):
                        j= i-1
                        j_3 = j-1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    result = ss.f_oneway(fieldvalues, fieldvalues1, fieldvalues3)
                                    self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                    self.ui_interfacewin.textEdit_result.append(str(result))
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 4组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_4.isChecked():
                if itemcount < 4:
                    QMessageBox.critical(self, 'Error', "至少要选择4列", QMessageBox.Ok)
                else:
                    for i in range(3, 4):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        result = ss.f_oneway(fieldvalues, fieldvalues1, fieldvalues3, fieldvalues4)
                                        self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                        self.ui_interfacewin.textEdit_result.append(str(result))
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 5组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_5.isChecked():
                if itemcount < 5:
                    QMessageBox.critical(self, 'Error', "至少要选择5列", QMessageBox.Ok)
                else:
                    for i in range(4, 5):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            result = ss.f_oneway(fieldvalues, fieldvalues1, fieldvalues3, fieldvalues4, fieldvalues5)
                                            self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                            self.ui_interfacewin.textEdit_result.append(str(result))
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 6组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_6.isChecked():
                if itemcount < 6:
                    QMessageBox.critical(self, 'Error', "至少要选择6列", QMessageBox.Ok)
                else:
                    for i in range(5, 6):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                result = ss.f_oneway(fieldvalues, fieldvalues1, fieldvalues3,
                                                                     fieldvalues4, fieldvalues5, fieldvalues6)
                                                self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                                self.ui_interfacewin.textEdit_result.append(str(result))
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6), QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 7组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_7.isChecked():
                if itemcount < 7:
                    QMessageBox.critical(self, 'Error', "至少要选择7列", QMessageBox.Ok)
                else:
                    for i in range(6, 7):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    result = ss.f_oneway(fieldvalues, fieldvalues1, fieldvalues3,
                                                                         fieldvalues4, fieldvalues5, fieldvalues6,
                                                                         fieldvalues7)
                                                    self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                                    self.ui_interfacewin.textEdit_result.append(str(result))
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7), QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6), QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 8组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_8.isChecked():
                if itemcount < 8:
                    QMessageBox.critical(self, 'Error', "至少要选择8列", QMessageBox.Ok)
                else:
                    for i in range(7, 8):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        j_8 = j_7 -1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name
                        field8 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_8).text()
                        fieldvalues8 = self.copydf[field8]
                        fieldvaluestype8 = fieldvalues8.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    if fieldvaluestype8 != 'object':
                                                        result = ss.f_oneway(fieldvalues, fieldvalues1, fieldvalues3,
                                                                             fieldvalues4, fieldvalues5, fieldvalues6,
                                                                             fieldvalues7, fieldvalues8)
                                                        self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                                        self.ui_interfacewin.textEdit_result.append(str(result))
                                                    else:
                                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field8),
                                                                             QMessageBox.Ok)
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7),
                                                                         QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6),
                                                                     QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 9组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_9.isChecked():
                if itemcount < 9:
                    QMessageBox.critical(self, 'Error', "至少要选择9列", QMessageBox.Ok)
                else:
                    for i in range(8, 9):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        j_8 = j_7 - 1
                        j_9 = j_8 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name
                        field8 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_8).text()
                        fieldvalues8 = self.copydf[field8]
                        fieldvaluestype8 = fieldvalues8.dtype.name
                        field9 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_9).text()
                        fieldvalues9 = self.copydf[field9]
                        fieldvaluestype9 = fieldvalues9.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    if fieldvaluestype8 != 'object':
                                                        if fieldvaluestype9 != 'object':
                                                            result = ss.f_oneway(fieldvalues, fieldvalues1,
                                                                                 fieldvalues3,
                                                                                 fieldvalues4, fieldvalues5,
                                                                                 fieldvalues6,
                                                                                 fieldvalues7, fieldvalues8, fieldvalues9)
                                                            self.ui_interfacewin.textEdit_result.append(
                                                                "1-way anova检验：")
                                                            self.ui_interfacewin.textEdit_result.append(str(result))
                                                        else:
                                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field9),
                                                                                 QMessageBox.Ok)
                                                    else:
                                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field8),
                                                                             QMessageBox.Ok)
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7),
                                                                         QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6),
                                                                     QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5),
                                                                 QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
             # 10组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_10.isChecked():
                if itemcount < 10:
                    QMessageBox.critical(self, 'Error', "至少要选择10列", QMessageBox.Ok)
                else:
                    for i in range(9, 10):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        j_8 = j_7 - 1
                        j_9 = j_8 - 1
                        j_10 = j_9 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name
                        field8 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_8).text()
                        fieldvalues8 = self.copydf[field8]
                        fieldvaluestype8 = fieldvalues8.dtype.name
                        field9 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_9).text()
                        fieldvalues9 = self.copydf[field9]
                        fieldvaluestype9 = fieldvalues9.dtype.name
                        field10 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_10).text()
                        fieldvalues10 = self.copydf[field10]
                        fieldvaluestype10 = fieldvalues10.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    if fieldvaluestype8 != 'object':
                                                        if fieldvaluestype9 != 'object':
                                                            if fieldvaluestype10 != 'object':
                                                                result = ss.f_oneway(fieldvalues, fieldvalues1,
                                                                                     fieldvalues3,
                                                                                     fieldvalues4, fieldvalues5,
                                                                                     fieldvalues6,
                                                                                     fieldvalues7, fieldvalues8,
                                                                                     fieldvalues9, fieldvalues10)
                                                                self.ui_interfacewin.textEdit_result.append(
                                                                    "1-way anova检验：")
                                                                self.ui_interfacewin.textEdit_result.append(str(result))
                                                            else:
                                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field10),
                                                                                     QMessageBox.Ok)
                                                        else:
                                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field9),
                                                                                 QMessageBox.Ok)
                                                    else:
                                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field8),
                                                                             QMessageBox.Ok)
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7),
                                                                         QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6),
                                                                     QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5),
                                                                 QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)

        #Kruskal-Wallis H方法
        if curtypeindex == 1:
            # 2组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_2.isChecked():
                if itemcount < 2:
                    QMessageBox.critical(self, 'Error', "至少要选择2列", QMessageBox.Ok)
                else:
                    for i in range(1, 2):
                        j = i - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                result = ss.kruskal(fieldvalues, fieldvalues1)
                                self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                self.ui_interfacewin.textEdit_result.append(str(result))
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 3组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_3.isChecked():
                if itemcount < 3:
                    QMessageBox.critical(self, 'Error', "至少要选择3列", QMessageBox.Ok)
                else:
                    for i in range(2, 3):
                        j = i - 1
                        j_3 = j - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    result = ss.kruskal(fieldvalues, fieldvalues1, fieldvalues3)
                                    self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                    self.ui_interfacewin.textEdit_result.append(str(result))
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 4组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_4.isChecked():
                if itemcount < 4:
                    QMessageBox.critical(self, 'Error', "至少要选择4列", QMessageBox.Ok)
                else:
                    for i in range(3, 4):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        result = ss.kruskal(fieldvalues, fieldvalues1, fieldvalues3, fieldvalues4)
                                        self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                        self.ui_interfacewin.textEdit_result.append(str(result))
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 5组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_5.isChecked():
                if itemcount < 5:
                    QMessageBox.critical(self, 'Error', "至少要选择5列", QMessageBox.Ok)
                else:
                    for i in range(4, 5):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            result = ss.kruskal(fieldvalues, fieldvalues1, fieldvalues3, fieldvalues4,
                                                                fieldvalues5)
                                            self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                            self.ui_interfacewin.textEdit_result.append(str(result))
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 6组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_6.isChecked():
                if itemcount < 6:
                    QMessageBox.critical(self, 'Error', "至少要选择6列", QMessageBox.Ok)
                else:
                    for i in range(5, 6):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                result = ss.kruskal(fieldvalues, fieldvalues1, fieldvalues3,
                                                                    fieldvalues4, fieldvalues5, fieldvalues6)
                                                self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                                self.ui_interfacewin.textEdit_result.append(str(result))
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6), QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 7组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_7.isChecked():
                if itemcount < 7:
                    QMessageBox.critical(self, 'Error', "至少要选择7列", QMessageBox.Ok)
                else:
                    for i in range(6, 7):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    result = ss.kruskal(fieldvalues, fieldvalues1, fieldvalues3,
                                                                        fieldvalues4, fieldvalues5, fieldvalues6,
                                                                        fieldvalues7)
                                                    self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                                    self.ui_interfacewin.textEdit_result.append(str(result))
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7),
                                                                         QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6), QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 8组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_8.isChecked():
                if itemcount < 8:
                    QMessageBox.critical(self, 'Error', "至少要选择8列", QMessageBox.Ok)
                else:
                    for i in range(7, 8):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        j_8 = j_7 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name
                        field8 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_8).text()
                        fieldvalues8 = self.copydf[field8]
                        fieldvaluestype8 = fieldvalues8.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    if fieldvaluestype8 != 'object':
                                                        result = ss.kruskal(fieldvalues, fieldvalues1, fieldvalues3,
                                                                            fieldvalues4, fieldvalues5, fieldvalues6,
                                                                            fieldvalues7, fieldvalues8)
                                                        self.ui_interfacewin.textEdit_result.append("1-way anova检验：")
                                                        self.ui_interfacewin.textEdit_result.append(str(result))
                                                    else:
                                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field8),
                                                                             QMessageBox.Ok)
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7),
                                                                         QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6),
                                                                     QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5), QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 9组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_9.isChecked():
                if itemcount < 9:
                    QMessageBox.critical(self, 'Error', "至少要选择9列", QMessageBox.Ok)
                else:
                    for i in range(8, 9):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        j_8 = j_7 - 1
                        j_9 = j_8 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name
                        field8 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_8).text()
                        fieldvalues8 = self.copydf[field8]
                        fieldvaluestype8 = fieldvalues8.dtype.name
                        field9 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_9).text()
                        fieldvalues9 = self.copydf[field9]
                        fieldvaluestype9 = fieldvalues9.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    if fieldvaluestype8 != 'object':
                                                        if fieldvaluestype9 != 'object':
                                                            result = ss.kruskal(fieldvalues, fieldvalues1,
                                                                                fieldvalues3,
                                                                                fieldvalues4, fieldvalues5,
                                                                                fieldvalues6,
                                                                                fieldvalues7, fieldvalues8,
                                                                                fieldvalues9)
                                                            self.ui_interfacewin.textEdit_result.append(
                                                                "1-way anova检验：")
                                                            self.ui_interfacewin.textEdit_result.append(str(result))
                                                        else:
                                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field9),
                                                                                 QMessageBox.Ok)
                                                    else:
                                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field8),
                                                                             QMessageBox.Ok)
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7),
                                                                         QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6),
                                                                     QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5),
                                                                 QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)
            # 10组数据
            if self.ui_interfacewin.radioButton_many_numbers_compare_10.isChecked():
                if itemcount < 10:
                    QMessageBox.critical(self, 'Error', "至少要选择10列", QMessageBox.Ok)
                else:
                    for i in range(9, 10):
                        j = i - 1
                        j_3 = j - 1
                        j_4 = j_3 - 1
                        j_5 = j_4 - 1
                        j_6 = j_5 - 1
                        j_7 = j_6 - 1
                        j_8 = j_7 - 1
                        j_9 = j_8 - 1
                        j_10 = j_9 - 1
                        field = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(i).text()
                        fieldvalues = self.copydf[field]
                        fieldvaluestype = fieldvalues.dtype.name
                        field1 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j).text()
                        fieldvalues1 = self.copydf[field1]
                        fieldvaluestype1 = fieldvalues.dtype.name
                        field3 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_3).text()
                        fieldvalues3 = self.copydf[field3]
                        fieldvaluestype3 = fieldvalues3.dtype.name
                        field4 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_4).text()
                        fieldvalues4 = self.copydf[field4]
                        fieldvaluestype4 = fieldvalues4.dtype.name
                        field5 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_5).text()
                        fieldvalues5 = self.copydf[field5]
                        fieldvaluestype5 = fieldvalues5.dtype.name
                        field6 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_6).text()
                        fieldvalues6 = self.copydf[field6]
                        fieldvaluestype6 = fieldvalues6.dtype.name
                        field7 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_7).text()
                        fieldvalues7 = self.copydf[field7]
                        fieldvaluestype7 = fieldvalues7.dtype.name
                        field8 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_8).text()
                        fieldvalues8 = self.copydf[field8]
                        fieldvaluestype8 = fieldvalues8.dtype.name
                        field9 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_9).text()
                        fieldvalues9 = self.copydf[field9]
                        fieldvaluestype9 = fieldvalues9.dtype.name
                        field10 = self.ui_interfacewin.listWidget_many_numbers_compare_2.item(j_10).text()
                        fieldvalues10 = self.copydf[field10]
                        fieldvaluestype10 = fieldvalues10.dtype.name

                        if fieldvaluestype != 'object':
                            if fieldvaluestype1 != 'object':
                                if fieldvaluestype3 != 'object':
                                    if fieldvaluestype4 != 'object':
                                        if fieldvaluestype5 != 'object':
                                            if fieldvaluestype6 != 'object':
                                                if fieldvaluestype7 != 'object':
                                                    if fieldvaluestype8 != 'object':
                                                        if fieldvaluestype9 != 'object':
                                                            if fieldvaluestype10 != 'object':
                                                                result = ss.kruskal(fieldvalues, fieldvalues1,
                                                                                    fieldvalues3,
                                                                                    fieldvalues4, fieldvalues5,
                                                                                    fieldvalues6,
                                                                                    fieldvalues7, fieldvalues8,
                                                                                    fieldvalues9, fieldvalues10)
                                                                self.ui_interfacewin.textEdit_result.append(
                                                                    "1-way anova检验：")
                                                                self.ui_interfacewin.textEdit_result.append(str(result))
                                                            else:
                                                                QMessageBox.critical(self, 'Error',
                                                                                     "%s列有字符" % (field10),
                                                                                     QMessageBox.Ok)
                                                        else:
                                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field9),
                                                                                 QMessageBox.Ok)
                                                    else:
                                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field8),
                                                                             QMessageBox.Ok)
                                                else:
                                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field7),
                                                                         QMessageBox.Ok)
                                            else:
                                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field6),
                                                                     QMessageBox.Ok)
                                        else:
                                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field5),
                                                                 QMessageBox.Ok)
                                    else:
                                        QMessageBox.critical(self, 'Error', "%s列有字符" % (field4), QMessageBox.Ok)
                                else:
                                    QMessageBox.critical(self, 'Error', "%s列有字符" % (field3), QMessageBox.Ok)
                            else:
                                QMessageBox.critical(self, 'Error', "%s列有字符" % (field1), QMessageBox.Ok)
                        else:
                            QMessageBox.critical(self, 'Error', "%s列有字符" % (field), QMessageBox.Ok)

    #相关性检验
    #添加col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_corr_test_itemClicked(self, item):
        itemname = item.text()
        self.ui_interfacewin.listWidget_corr_test_2.addItem(itemname)

    #删除col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_corr_test_2_itemClicked(self, item):
        row = self.ui_interfacewin.listWidget_corr_test_2.currentIndex().row()
        self.ui_interfacewin.listWidget_corr_test_2.takeItem(row)

    #检验
    @pyqtSlot()
    def on_corr_test_Btn_clicked(self):
        curtype =  self.ui_interfacewin.comboBox_corr_test.currentIndex()
        itemcount = self.ui_interfacewin.listWidget_corr_test_2.count()
        itemcount2 = itemcount-1
        if curtype == 0:
            for i in range(0, itemcount2):
                j = i+1
                field = self.ui_interfacewin.listWidget_corr_test_2.item(i).text()
                field2 = self.ui_interfacewin.listWidget_corr_test_2.item(j).text()
                fieldvalues = self.copydf[field]
                fieldvalues2 = self.copydf[field2]
                fieldvaluestype = fieldvalues.dtype.name
                fieldvaluestype2 = fieldvalues2.dtype.name
                if fieldvaluestype != 'object':
                    if fieldvaluestype2 != 'object':
                        result = ss.pearsonr(fieldvalues, fieldvalues)
                        self.ui_interfacewin.textEdit_result.append("%s列和%s列的Pearson相关系数:"%(field, field2))
                        self.ui_interfacewin.textEdit_result.append(str(result))
                    else:
                        QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
        if curtype == 1:
            for i in range(0, itemcount2):
                j = i+1
                field = self.ui_interfacewin.listWidget_corr_test_2.item(i).text()
                field2 = self.ui_interfacewin.listWidget_corr_test_2.item(j).text()
                fieldvalues = self.copydf[field]
                fieldvalues2 = self.copydf[field2]
                fieldvaluestype = fieldvalues.dtype.name
                fieldvaluestype2 = fieldvalues2.dtype.name
                if fieldvaluestype != 'object':
                    if fieldvaluestype2 != 'object':
                        result = ss.spearmanr(fieldvalues, fieldvalues)
                        self.ui_interfacewin.textEdit_result.append("%s列和%s列的Spearman相关系数"%(field, field2))
                        self.ui_interfacewin.textEdit_result.append(str(result))
                    else:
                        QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
    #二元值与连续值之间的关系
    #添加col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_Binary_and_continuous_values_test_itemClicked(self, item):
        itemname = item.text()
        self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.addItem(itemname)
    #删除col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_Binary_and_continuous_values_test_2_itemClicked(self, item):
        row = self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.currentIndex().row()
        self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.takeItem(row)

    #检验
    @pyqtSlot()
    def on_Binary_and_continuous_values_test_Btn_clicked(self):
        curtype = self.ui_interfacewin.comboBox_Binary_and_continuous_values_test.currentIndex()
        itemcount = self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.count()
        itemcount2 = itemcount - 1
        if curtype == 0:
            for i in range(0, itemcount2):
                j = i + 1
                field = self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.item(i).text()
                field2 = self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.item(j).text()
                fieldvalues = self.copydf[field]
                fieldvalues2 = self.copydf[field2]
                fieldvaluestype = fieldvalues.dtype.name
                fieldvaluestype2 = fieldvalues2.dtype.name
                if fieldvaluestype != 'object':
                    if fieldvaluestype2 != 'object':
                        result = ss.mstats.pointbiserialr(fieldvalues, fieldvalues)
                        self.ui_interfacewin.textEdit_result.append("%s列和%s列的二元值和连续值之间的关系:" % (field, field2))
                        self.ui_interfacewin.textEdit_result.append(str(result))
                    else:
                        QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
        if curtype == 1:
            for i in range(0, itemcount2):
                j = i + 1
                field = self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.item(i).text()
                field2 = self.ui_interfacewin.listWidget_Binary_and_continuous_values_test_2.item(j).text()
                fieldvalues = self.copydf[field]
                fieldvalues2 = self.copydf[field2]
                fieldvaluestype = fieldvalues.dtype.name
                fieldvaluestype2 = fieldvalues2.dtype.name
                if fieldvaluestype != 'object':
                    if fieldvaluestype2 != 'object':
                        result = ss.mstats.kendalltau(fieldvalues, fieldvalues)
                        self.ui_interfacewin.textEdit_result.append("%s列和%s列的二元值和连续值之间的关系:" % (field, field2))
                        self.ui_interfacewin.textEdit_result.append(str(result))
                    else:
                        QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
                else:
                    QMessageBox.critical(self, "Error", "%s列有字符", QMessageBox.Ok)
    # 多因子数据分析模块************************************************************************************************
   #数据清洗
    #条件选择框变化
    @pyqtSlot(int)
    def on_comboBox_detection_outlier_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(7)
        if index == 1:
            self.ui_interfacewin.stackedWidget_exploratory_data_analysis.setCurrentIndex(6)
    #异常值处理
    #添加条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_detection_outlier_itemClicked(self, item):
        itemname = item.text()
        curchioce = self.ui_interfacewin.checkBox_detection_outlier.isChecked()
        if curchioce:
            self.ui_interfacewin.listWidget_detection_outlier_3.clear()
            field = self.copydf[itemname].unique()
            str2 = field.tolist()
            new_str2 = [str(x) for x in str2]
            self.ui_interfacewin.listWidget_detection_outlier_3.addItems(new_str2)
        else:
            self.ui_interfacewin.listWidget_detection_outlier_2.addItem(itemname)
    #删除条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_detection_outlier_2_itemClicked(self, item):
        row = self.ui_interfacewin.listWidget_detection_outlier_2.currentIndex().row()
        self.ui_interfacewin.listWidget_detection_outlier_2.takeItem(row)
    #异常值检测
    @pyqtSlot()
    def on_detection_outlier_Btn_clicked(self):
        itemcount = self.ui_interfacewin.listWidget_detection_outlier_2.count()
        curtype = self.ui_interfacewin.comboBox_detection_outlier_4.currentIndex()
        #无离散异常值检测
        #连续异常值检测
        if curtype == 0:
            for i in range(0, itemcount):
                field = self.ui_interfacewin.listWidget_detection_outlier_2.item(i).text()
                fieldvalues = self.copydf[field]
                fieldvaluestype = fieldvalues.dtype.name

                if fieldvaluestype != 'object':
                    q_low = fieldvalues.quantile(q=0.25)
                    q_high = fieldvalues.quantile(q=0.75)
                    k=1.5
                    q_interval = q_high - q_low
                    le_s = fieldvalues[fieldvalues > q_high+k*q_interval]
                    le_s2 = fieldvalues[fieldvalues < q_low-k*q_interval]

                    if len(le_s) != 0:
                        self.ui_interfacewin.textEdit_result.append("%s列异常值："%(field))
                        self.ui_interfacewin.textEdit_result.append(str(le_s))
                    if len(le_s2) != 0:
                        self.ui_interfacewin.textEdit_result.append("%s列异常值："%(field))
                        self.ui_interfacewin.textEdit_result.append(str(le_s2))
                    if len(le_s2) == 0 and len(le_s) == 0:
                            self.ui_interfacewin.textEdit_result.append("%s列无异常值："%(field))
        #知识异常值检测
        curtype_2 = self.ui_interfacewin.comboBox_detection_outlier_3.currentIndex()
        if curtype == 1:
                value = self.ui_interfacewin.doubleSpinBox_detection_outlier.value()
                value2 = self.ui_interfacewin.doubleSpinBox_detection_outlier_2.value()
                itemcount = self.ui_interfacewin.listWidget_detection_outlier_2.count()
                curindex = self.ui_interfacewin.comboBox_detection_outlier.currentIndex()
                for i in range(0, itemcount):
                    field = self.ui_interfacewin.listWidget_detection_outlier_2.item(i).text()
                    fieldvalues = self.copydf[field]
                    fieldvaluestype = fieldvalues.dtype.name

                    if fieldvaluestype != 'object':
                        if curtype_2 == 0:
                            le_s = fieldvalues[fieldvalues < value]
                            le_s2 = fieldvalues[fieldvalues > value2]
                        if curtype_2 == 1:
                            le_s = self.copydf[fieldvalues >= value][fieldvalues < value2]

                        if curtype_2 == 2:
                            le_s = fieldvalues[fieldvalues == value]

                        if curtype_2 == 0:
                            if len(le_s) == 0:
                                if len(le_s2) == 0:
                                    self.ui_interfacewin.textEdit_result.append("%s列无异常值：" % (field))
                                else:
                                    self.ui_interfacewin.textEdit_result.append("%s列异常值1：" % (field))
                                    self.ui_interfacewin.textEdit_result.append(str(le_s2))

                            else:
                                self.ui_interfacewin.textEdit_result.append("%s列异常值2：" % (field))
                                self.ui_interfacewin.textEdit_result.append(str(le_s))
                                pd.Series(le_s).to_csv('/home/niangu/桌面/比赛文件/文件2(怠速).csv')
                                print("CCCCCCCCC")
                        else:
                            if len(le_s) != 0:
                                self.ui_interfacewin.textEdit_result.append("%s列异常值：" % (field))
                                self.ui_interfacewin.textEdit_result.append(str(le_s))
                                pd.DataFrame(le_s).to_csv('/home/niangu/桌面/比赛文件/文件2(怠速).csv')
                                print("CCCCCCCCC")
                            else:
                                self.ui_interfacewin.textEdit_result.append("%s列无异常值：" % (field))
        if curtype == 2:
            #离散异常值检测
            field = self.ui_interfacewin.listWidget_detection_outlier.currentItem().text()
            fieldvalues = self.ui_interfacewin.listWidget_detection_outlier_3.currentItem().text()
            if is_number(fieldvalues):
                fieldvalues = float(fieldvalues)

            fieldname = self.copydf[self.copydf[field]==fieldvalues]
            self.ui_interfacewin.textEdit_result.append("%s列离散异常值:%s:%s" % (field, fieldvalues, fieldname))

    #异常值处理
    @pyqtSlot()
    def on_detection_outlier_Btn_2_clicked(self):
        itemcount = self.ui_interfacewin.listWidget_detection_outlier_2.count()
        datatype = self.ui_interfacewin.comboBox_detection_outlier_4.currentIndex()
        knowdatatype = self.ui_interfacewin.comboBox_detection_outlier_3.currentIndex()
        opeatortype = self.ui_interfacewin.comboBox_detection_outlier_2.currentIndex()
        #知识异常值
        value = self.ui_interfacewin.doubleSpinBox_detection_outlier.value()
        value2 = self.ui_interfacewin.doubleSpinBox_detection_outlier_2.value()
        curindex = self.ui_interfacewin.comboBox_detection_outlier.currentIndex()
        curtype_2 = self.ui_interfacewin.comboBox_detection_outlier_3.currentIndex()

        if datatype == 0 or datatype == 1:
            for i in range(0, itemcount):
                field = self.ui_interfacewin.listWidget_detection_outlier_2.item(i).text()
                fieldvalues = self.copydf[field]
                fieldvaluestype = fieldvalues.dtype.name
                if fieldvaluestype != 'object':
                    #连续异常值
                    if datatype == 0:
                        q_low = fieldvalues.quantile(q=0.25)
                        q_high = fieldvalues.quantile(q=0.75)
                        k = 1.5
                        q_interval = q_high - q_low
                        le_s = fieldvalues[fieldvalues > q_high + k * q_interval]
                        le_s2 = fieldvalues[fieldvalues < q_low - k * q_interval]
                        fieldvalues.replace(le_s, np.nan, inplace=True)
                        fieldvalues.replace(le_s2, np.nan, inplace=True)
                        rowcount = len(le_s)+len(le_s2)
                    #知识异常值
                    if datatype == 1:
                        if curtype_2 == 0:
                            le_s = fieldvalues[fieldvalues < value]
                            le_s2 = fieldvalues[fieldvalues > value2]
                            fieldvalues.replace(le_s, np.nan, inplace=True)
                            fieldvalues.replace(le_s2, np.nan, inplace=True)
                            rowcount = len(le_s) + len(le_s2)
                        if curtype_2 == 1:
                            #le_s = fieldvalues[fieldvalues > value][fieldvalues < value2]
                            #fieldvalues.replace(le_s, np.nan, inplace=True)
                            #rowcount = len(le_s)
                            print('AAAAA')
                        if curtype_2 == 2:
                            le_s = fieldvalues[fieldvalues == value]
                            fieldvalues.replace(le_s, np.nan, inplace=True)
                            rowcount = len(le_s)

                    if opeatortype == 0:
                        mean = fieldvalues.mean()
                        fieldvalues.fillna(mean, inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列平均值：%s替换%s行异常值" % (field, mean, rowcount))
                    if opeatortype == 1:
                        median = fieldvalues.median()
                        fieldvalues.fillna(median, inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列中位数：%s替换%s行异常值" % (field, median, rowcount))
                    if opeatortype == 2:
                        mode1 = list(fieldvalues.mode())
                        fieldvalues.fillna(mode1[0], inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列众数：%s替换%s行异常值" % (field, mode1, rowcount))
                    if opeatortype == 3:
                        fieldvalues.fillna(method='ffill', inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列前项替换%s行异常值" % (field, rowcount))
                    if opeatortype == 4:
                        fieldvalues.fillna(method='bfill', inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列后项替换%s行异常值" % (field, rowcount))
                    if opeatortype == 5:
                        quantile = fieldvalues.quantile(0.25)
                        fieldvalues.fillna(quantile, inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列上四分位数：%s替换%s行异常值" % (field, quantile, rowcount))
                    if opeatortype == 6:
                        quantile = fieldvalues.quantile(0.5)
                        fieldvalues.fillna(quantile, inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列中四分位数：%s替换%s行异常值" % (field, quantile, rowcount))
                    if opeatortype == 7:
                        quantile = fieldvalues.quantile(0.75)
                        fieldvalues.fillna(quantile, inplace=True)
                        self.ui_interfacewin.textEdit_result.append(
                            "%s列下四分位数：%s替换%s行异常值" % (field, quantile, rowcount))
                    if opeatortype == 8:
                        field2 = pd.Series(field)
                        field2.to_csv('/home/niangu/桌面/比赛文件/文件2(怠速).csv')
                        #fieldvalues.dropna(axis=0, how='any')
                        #self.ui_interfacewin.textEdit_result.append(
                            #"%s列删除%s行异常值" % (field, rowcount))
                        print("BBBBBBBBB")
        # 离散异常值
        if datatype == 2:
            curopeatortype2 = self.ui_interfacewin.comboBox_detection_outlier_5.currentIndex()
            field2 = self.ui_interfacewin.listWidget_detection_outlier.currentItem().text()
            fieldvalues2 = self.copydf[field2]
            curitem = self.ui_interfacewin.listWidget_detection_outlier_3.currentItem().text()
            le_s = fieldvalues2[fieldvalues2 == curitem]
            fieldvalues2.replace(le_s, np.nan, inplace=True)
            rowcount = len(le_s)
            if curopeatortype2 == 0:
                mode1 = list(fieldvalues2.mode())
                fieldvalues2.fillna(mode1[0], inplace=True)
                self.ui_interfacewin.textEdit_result.append(
                    "%s列众数：%s替换%s行异常值" % (field2, mode1, rowcount))
            if curopeatortype2 == 1:
                fieldvalues2.fillna(method='ffill', inplace=True)
                self.ui_interfacewin.textEdit_result.append(
                    "%s列前项替换%s行异常值" % (field2, rowcount))
            if curopeatortype2 == 2:
                fieldvalues2.fillna(method='bfill', inplace=True)
                self.ui_interfacewin.textEdit_result.append(
                    "%s列后项替换%s行异常值" % (field2, rowcount))
            if curopeatortype2 == 4:
                # fieldvalues.dropna(axis=1, how='any', inplace=True)#how='all'当列全部为缺失值时删除
                self.copydf.drop([field2], axis=1, inplace=True)
                # self.copydf.drop(columns=[field])
                columns = self.copydf.columns
                self.ui_interfacewin.listWidget_detection_outlier.clear()
                self.ui_interfacewin.listWidget_detection_outlier.addItems(columns)
                self.ui_interfacewin.listWidget_detection_outlier_3.clear()

                self.ui_interfacewin.textEdit_result.append("删除%s列" % (field2))
            if curopeatortype2 == 3:
                index = fieldvalues2[fieldvalues2.isnull()==True].index
                self.copydf = self.copydf.dropna(subset=[field2])

                self.ui_interfacewin.textEdit_result.append("删除%s" % (index))


    #空值检测
    #添加条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_null_values_itemClicked(self, item):
        curchioce = self.ui_interfacewin.checkBox_null_values.isChecked()
        itemtext = item.text()
        if curchioce:
            self.ui_interfacewin.listWidget_null_values_3.addItem(itemtext)
        else:
            self.ui_interfacewin.listWidget_null_values_2.addItem(itemtext)
    #删除条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_null_values_2_itemClicked(self, item):
        row = self.ui_interfacewin.listWidget_null_values_2.currentIndex().row()
        self.ui_interfacewin.listWidget_null_values_2.takeItem(row)
    #删除条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_null_values_3_itemClicked(self, item):
        row = self.ui_interfacewin.listWidget_null_values_3.currentIndex().row()
        self.ui_interfacewin.listWidget_null_values_3.takeItem(row)
    #检测
    @pyqtSlot()
    def on_null_values_Btn_clicked(self):
        itemcount = self.ui_interfacewin.listWidget_null_values_2.count()
        for i in range(0, itemcount):
            field = self.ui_interfacewin.listWidget_null_values_2.item(i).text()
            fieldvalues = self.copydf[field]
            fieldnull = fieldvalues[fieldvalues.isnull()]
            nullcount = len(fieldnull)
            fieldcount = len(fieldvalues)
            if nullcount != 0:
                self.ui_interfacewin.textEdit_result.append("%s列：%s行" % (field, fieldcount))
                self.ui_interfacewin.textEdit_result.append("%s列缺失值:%s个：" % (field, nullcount))

                self.ui_interfacewin.textEdit_result.append(str(fieldnull))
            if nullcount == 0:
                self.ui_interfacewin.textEdit_result.append("%s列无缺失值：" % (field))

    #空值处理
    @pyqtSlot()
    def on_null_values_Btn_2_clicked(self):
        itemcount = self.ui_interfacewin.listWidget_null_values_2.count()
        curtype = self.ui_interfacewin.comboBox_null_values.currentIndex()
        limit_direction = self.ui_interfacewin.comboBox_null_values_2.currentText()
        datatype = self.ui_interfacewin.comboBox_null_values_3.currentIndex()
        if datatype == 0:
            for i in range(0, itemcount):
                field = self.ui_interfacewin.listWidget_null_values_2.item(i).text()
                fieldvalues = self.copydf[field]
                fieldvaluestype = fieldvalues.dtype.name
                if fieldvaluestype != 'object':
                    nullcount = np.isnan(fieldvalues).sum()
                    if nullcount == 0:
                        self.ui_interfacewin.textEdit_result.append("%s列无空值：" % (field))
                    if nullcount != 0:
                        if curtype == 0:
                            #fieldvalues.dropna(axis=1, how='any', inplace=True)#how='all'当列全部为缺失值时删除
                            self.copydf.drop([field], axis=1, inplace=True)
                            #self.copydf.drop(columns=[field])
                            self.refresh()
                            self.ui_interfacewin.textEdit_result.append("删除%s列"%(field))
                        if curtype == 1:
                            index = fieldvalues[fieldvalues.isnull() == True].index
                            #fieldvalues.dropna(axis=0, how='any', inplace=True)
                            self.copydf = self.copydf.dropna(subset=[field])
                            #self.refresh()
                            self.ui_interfacewin.textEdit_result.append("删除%s行" % (index))
                        if curtype == 2:
                            mean = fieldvalues.mean()
                            fieldvalues.fillna(mean, inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列平均值：%s填充%s行空值" % (field, mean, nullcount))
                        if curtype == 3:
                            median = fieldvalues.median()
                            fieldvalues.fillna(median, inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列中位数：%s填充%s行空值" % (field, median, nullcount))
                        if curtype == 4:
                            mode1 = list(fieldvalues.mode())
                            fieldvalues.fillna(mode1[0], inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列众数：%s填充%s行空值" % (field, mode1, nullcount))
                        if curtype == 5:
                            fieldvalues.fillna(method='ffill', inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列前项填充完毕"%(field))
                        if curtype == 6:
                            fieldvalues.fillna(method='bfill', inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列后项填充完毕"%(field))
                        if curtype == 7:
                            quantile = fieldvalues.quantile(0.25)
                            fieldvalues.fillna(quantile, inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列上四分位数：%s填充%s行空值" % (field, quantile, nullcount))
                        if curtype == 8:
                            quantile2 = fieldvalues.quantile(0.5)
                            fieldvalues.fillna(quantile2, inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列中四分位数：%s填充%s行空值" % (field, quantile2, nullcount))
                        if curtype == 9:
                            quantile3 = fieldvalues.quantile(0.75)
                            fieldvalues.fillna(quantile3, inplace=True)
                            self.ui_interfacewin.textEdit_result.append("%s列下四分位数：%s填充%s行空值" % (field, quantile3, nullcount))
                        if curtype == 10:
                            from scipy.interpolate import interp1d
                            count = self.ui_interfacewin.listWidget_null_values_3.count()
                            if count == 1:
                                ytext = self.ui_interfacewin.listWidget_null_values_3.item(0).text()
                                yvalue = self.copydf[ytext]
                                linearinsvalue = interp1d(yvalue, fieldvalues, kind='linear')
                                nullindex = fieldvalues[fieldvalues.isnull()==True].index
                                yvalue2 = yvalue[nullindex]
                                result = pd.Series(linearinsvalue(yvalue2))
                                for index in result.index:
                                    #fieldvalues.fillna(value=result.loc[index], inplace=True, limit=1)
                                    fieldvalues.fillna(value=result.loc[index])
                                nullcount2 = np.isnan(fieldvalues).sum()
                                if nullcount2 == 0:
                                    fieldvalues2 = self.copydf[field]
                                    for index in result.index:
                                         fieldvalues2.fillna(value=result.loc[index], inplace=True, limit=1)
                                    self.ui_interfacewin.textEdit_result.append("%s列与%s列线性插值:%s,%s个" % (field, ytext, result, nullcount))
                                else:
                                    ok = QMessageBox.critical(self, "Error", "未全部匹配是否取消", QMessageBox.Ok, QMessageBox.Cancel)
                                    if ok == QMessageBox.Ok:
                                        self.ui_interfacewin.textEdit_result.append(
                                            "%s列线性插值取消" % (field))
                                    else:
                                        fieldvalues2 = self.copydf[field]
                                        for index in result.index:
                                            fieldvalues2.fillna(value=result.loc[index], inplace=True, limit=1)
                                        self.ui_interfacewin.textEdit_result.append("%s列与%s列线性插值:%s,%s个" % (field, ytext, result, nullcount))

                            if count > 1 and count == itemcount:
                                for j in range(0, count):
                                    if i == j:
                                        ytext = self.ui_interfacewin.listWidget_null_values_3.item(j).text()
                                        yvalue = self.copydf[ytext]
                                        linearinsvalue = interp1d(yvalue, fieldvalues, kind='linear')
                                        nullindex = fieldvalues[fieldvalues.isnull() == True].index
                                        yvalue2 = yvalue[nullindex]
                                        result = pd.Series(linearinsvalue(yvalue2))
                                        for index in result.index:
                                            # fieldvalues.fillna(value=result.loc[index], inplace=True, limit=1)
                                            fieldvalues.fillna(value=result.loc[index])
                                        nullcount2 = np.isnan(fieldvalues).sum()
                                        if nullcount2 == 0:
                                            fieldvalues2 = self.copydf[field]
                                            for index in result.index:
                                                fieldvalues2.fillna(value=result.loc[index], inplace=True, limit=1)
                                            self.ui_interfacewin.textEdit_result.append(
                                                "%s列与%s列线性插值:%s,%s个" % (field, ytext, result, nullcount))
                                        else:
                                            ok = QMessageBox.critical(self, "Error", "未全部匹配是否取消", QMessageBox.Ok,
                                                                      QMessageBox.Cancel)
                                            if ok == QMessageBox.Ok:
                                                self.ui_interfacewin.textEdit_result.append(
                                                    "%s列线性插值取消" % (field))
                                            else:
                                                fieldvalues2 = self.copydf[field]
                                                for index in result.index:
                                                    fieldvalues2.fillna(value=result.loc[index], inplace=True, limit=1)
                                                self.ui_interfacewin.textEdit_result.append(
                                                    "%s列与%s列线性插值:%s,%s个" % (field, ytext, result, nullcount))
                        if curtype == 11:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='linear', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%s列线性填充:%s, %s个"%(field, result, resultlen))
                        if curtype == 12:
                            data = self.copydf
                            i = field
                            result = []
                            for j in range(len(data)):
                                if (data[i].isnull())[j]:
                                    data[i][j] = self.ployinter(data[i], j)
                                    result.append(data[i][j])

                            self.ui_interfacewin.textEdit_result.append(
                                    "%s列拉格朗日插值:%s,共%s个" % (field, result, nullcount))
                        if curtype == 13:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='polynomial', order=5, inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%s多项式填充:%s, %s个" % (field, result, resultlen))
                        if curtype == 14:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='spline', order=5, inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%s样条插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 15:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='quadratic', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%s3 阶B样条曲线插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 16:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='cubic', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%s2 阶B样条曲线插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 17:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='barycentric', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%sbarycentric插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 18:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='nearest', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%snearest插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 19:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='zero', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%szero插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 20:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='pad', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%spad插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 21:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='index', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%sindex插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 22:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='slinear', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%sslinear插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 23:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='krogh', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%skrogh插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 24:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='piecewise_polynomial', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%spiecewise_polynomial插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 25:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='pchip', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%spchip插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 26:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='akima', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%sakima插值:%s, %s个" % (field, result, resultlen))
                        if curtype == 27:
                            nullindex = fieldvalues[fieldvalues.isnull() == True].index
                            fieldvalues.interpolate(method='from_derivatives', inplace=True, limit_direction=limit_direction)
                            result = fieldvalues[nullindex]
                            resultlen = len(result)
                            self.ui_interfacewin.textEdit_result.append("%sfrom_derivatives插值:%s, %s个" % (field, result, resultlen))
        if datatype == 1:
            curdatetype = self.ui_interfacewin.comboBox_null_values_4.currentIndex()
            for i in range(0, itemcount):
                field = self.ui_interfacewin.listWidget_null_values_2.item(i).text()
                fieldvalues = self.copydf[field]
                fieldnull = fieldvalues[fieldvalues.isnull()]
                nullcount = len(fieldnull)
                if nullcount == 0:
                    self.ui_interfacewin.textEdit_result.append("%s列无空值：" % (field))
                if nullcount != 0:
                    if curdatetype == 0:
                         nullindex = fieldvalues[fieldvalues.isnull() == True].index
                         fieldvalues.interpolate(method='time', inplace=True, limit_direction=limit_direction)
                         result = fieldvalues[nullindex]
                         resultlen = len(result)
                         self.ui_interfacewin.textEdit_result.append("%s列插值:%s, %s个" % (field, result, resultlen))
                    if curdatetype == 1:
                         fieldvalues.fillna(method='ffill', inplace=True)
                         self.ui_interfacewin.textEdit_result.append("%s列前项填充完毕" % (field))
                    if curdatetype == 2:
                         fieldvalues.fillna(method='bfill', inplace=True)
                         self.ui_interfacewin.textEdit_result.append("%s列后项填充完毕" % (field))
                    if curdatetype == 3:
                         # fieldvalues.dropna(axis=1, how='any', inplace=True)#how='all'当列全部为缺失值时删除
                         self.copydf.drop([field], axis=1, inplace=True)
                         # self.copydf.drop(columns=[field])
                         self.refresh()
                         self.ui_interfacewin.textEdit_result.append("删除%s列" % (field))
                    if curdatetype == 4:
                         index = fieldvalues[fieldvalues.isnull() == True].index
                         fieldvalues.dropna(axis=0, how='any', inplace=True)
                         self.refresh()
                         self.ui_interfacewin.textEdit_result.append("删除%s行" % (index))
        if datatype == 2:
            curdatetype = self.ui_interfacewin.comboBox_null_values_5.currentIndex()
            for i in range(0, itemcount):
                field = self.ui_interfacewin.listWidget_null_values_2.item(i).text()
                fieldvalues = self.copydf[field]
                fieldnull = fieldvalues[fieldvalues.isnull()]
                nullcount = len(fieldnull)
                if nullcount == 0:
                    self.ui_interfacewin.textEdit_result.append("%s列无空值：" % (field))
                if nullcount != 0:
                    if curdatetype == 0:
                        fieldvalues.fillna(method='ffill', inplace=True)
                        self.ui_interfacewin.textEdit_result.append("%s列前项填充完毕" % (field))
                    if curdatetype == 1:
                        fieldvalues.fillna(method='bfill', inplace=True)
                        self.ui_interfacewin.textEdit_result.append("%s列后项填充完毕" % (field))
                    if curdatetype == 2:
                        # fieldvalues.dropna(axis=1, how='any', inplace=True)#how='all'当列全部为缺失值时删除
                        self.copydf.drop([field], axis=1, inplace=True)
                        # self.copydf.drop(columns=[field])
                        self.refresh()
                        self.ui_interfacewin.textEdit_result.append("删除%s列" % (field))
                    if curdatetype == 3:
                        index = fieldvalues[fieldvalues.isnull() == True].index
                        fieldvalues.dropna(axis=0, how='any', inplace=True)
                        self.refresh()
                        self.ui_interfacewin.textEdit_result.append("删除%s行" % (index))
                    if curdatetype == 4:
                        mode1 = list(fieldvalues.mode())
                        fieldvalues.fillna(mode1[0], inplace=True)
                        self.ui_interfacewin.textEdit_result.append("%s列众数：%s填充%s行空值" % (field, mode1, nullcount))
    #去重
    @pyqtSlot()
    def on_dupplication_Btn_clicked(self):
        result = self.copydf.drop_duplicates(inplace=True)
        self.ui_interfacewin.textEdit_result.append("去重：%s"%(result))

    #线性回归
    #添加项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_linear_regression_itemClicked(self, item):
        itemtext = item.text()
        self.ui_interfacewin.listWidget_linear_regression_2.addItem(itemtext)
    #删除col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_linear_regression_2_itemClicked(self, item):
        currow = self.ui_interfacewin.listWidget_linear_regression_2.currentIndex().row()
        self.ui_interfacewin.listWidget_linear_regression_2.takeItem(currow)
    #拟合
    @pyqtSlot()
    def on_linear_regression_Btn_clicked(self):
        fit_intercept = False
        normalize = False
        copy_X = False
        n_jobs = None
        if self.ui_interfacewin.checkBox_linear_regression.isChecked():
            fit_intercept = True
        if self.ui_interfacewin.checkBox_linear_regression_2.isChecked():
            normalize = True
        if self.ui_interfacewin.checkBox_linear_regression_3.isChecked():
            copy_X = True
        if self.ui_interfacewin.checkBox_linear_regression_4.isChecked():
            n_jobs = self.ui_interfacewin.spinBox_linear_regression.value()
        fieldcount = self.ui_interfacewin.listWidget_linear_regression_2.count()
        fieldcount2 = fieldcount - 1

        if fieldcount > 2:
            for i in range(0, fieldcount2):
                field = self.ui_interfacewin.listWidget_linear_regression_2.item(i).text()
                j = i+1
                field2 = self.ui_interfacewin.listWidget_linear_regression_2.item(j).text()
                fieldvalues = np.array(self.copydf[field]).reshape(-1, 1)
                fieldvalues2 = np.array(self.copydf[field2]).reshape(-1, 1)
                linear_reg = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
                reg = linear_reg.fit(fieldvalues, fieldvalues2)
                y_pred = reg.predict(fieldvalues)
                self.ui_interfacewin.textEdit_result.append("%s列和%s列线性拟合(线性回归)：" % (field, field2))
                self.ui_interfacewin.textEdit_result.append("回归系数：%s"%(reg.coef_))
                self.ui_interfacewin.textEdit_result.append(("截距%s"%(reg.intercept_)))
                self.ui_interfacewin.textEdit_result.append("决定系数(方差分数)R^2:%s,   %s" % (reg.score(fieldvalues, fieldvalues2), r2_score(fieldvalues2, y_pred)))
                self.ui_interfacewin.textEdit_result.append("均方误差：%s" % (mean_squared_error(fieldvalues2, y_pred)))
                if self.ui_interfacewin.checkBox_linear_regression_5.isChecked():
                    self.ui_interfacewin.textEdit_result.append("%s列输入拟合结果：\n%s" % (field, y_pred))
                plt.figure()
                plt.plot(fieldvalues.reshape(1, -1)[0], fieldvalues2.reshape(1, -1)[0], "r*")
                plt.plot(fieldvalues.reshape(1, -1)[0], y_pred.reshape(1, -1)[0])
                plt.show()


    #PCA降维
    #添加项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_PCA_itemClicked(self, item):
        itemtext = item.text()
        self.ui_interfacewin.listWidget_PCA_2.addItem(itemtext)
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_PCA_2_itemClicked(self, item):
        currow = self.ui_interfacewin.listWidget_PCA_2.currentIndex().row()
        self.ui_interfacewin.listWidget_PCA_2.takeItem(currow)
    #n_components参数框变化设置
    @pyqtSlot(int)
    def on_checkBox_PCA_n_components_double_stateChanged(self, state):
        if state:
            self.ui_interfacewin.checkBox_PCA_n_components_mle.setEnabled(False)
            self.ui_interfacewin.checkBox_PCA_n_components_int.setEnabled(False)
            self.ui_interfacewin.spinBoxn_PCA_n_components_int.setEnabled(False)
        else:
            self.ui_interfacewin.checkBox_PCA_n_components_mle.setEnabled(True)
            self.ui_interfacewin.checkBox_PCA_n_components_int.setEnabled(True)
            self.ui_interfacewin.spinBoxn_PCA_n_components_int.setEnabled(True)
    @pyqtSlot(int)
    def on_checkBox_PCA_n_components_int_stateChanged(self, state):
        if state:
            self.ui_interfacewin.checkBox_PCA_n_components_mle.setEnabled(False)
            self.ui_interfacewin.checkBox_PCA_n_components_double.setEnabled(False)
            self.ui_interfacewin.doubleSpinBox_PCA_n_components_double.setEnabled(False)
        else:
            self.ui_interfacewin.checkBox_PCA_n_components_mle.setEnabled(True)
            self.ui_interfacewin.checkBox_PCA_n_components_double.setEnabled(True)
            self.ui_interfacewin.doubleSpinBox_PCA_n_components_double.setEnabled(True)
    @pyqtSlot(int)
    def on_checkBox_PCA_n_components_mle_stateChanged(self, state):
        if state:
            self.ui_interfacewin.checkBox_PCA_n_components_double.setEnabled(False)
            self.ui_interfacewin.doubleSpinBox_PCA_n_components_double.setEnabled(False)
            self.ui_interfacewin.checkBox_PCA_n_components_int.setEnabled(False)
            self.ui_interfacewin.spinBoxn_PCA_n_components_int.setEnabled(False)
        else:
            self.ui_interfacewin.checkBox_PCA_n_components_double.setEnabled(True)
            self.ui_interfacewin.doubleSpinBox_PCA_n_components_double.setEnabled(True)
            self.ui_interfacewin.checkBox_PCA_n_components_int.setEnabled(True)
            self.ui_interfacewin.spinBoxn_PCA_n_components_int.setEnabled(True)
    #random_satte参数框变化设置
    @pyqtSlot(int)
    def on_checkBox_PCA_RandomState_stateChanged(self, state):
        if state:
            self.ui_interfacewin.checkBox_PCA_random_state_int.setEnabled(False)
            self.ui_interfacewin.spinBox_PCA_random_state_int.setEnabled(False)
        else:
            self.ui_interfacewin.checkBox_PCA_random_state_int.setEnabled(True)
            self.ui_interfacewin.spinBox_PCA_random_state_int.setEnabled(True)
    @pyqtSlot(int)
    def on_checkBox_PCA_random_state_int_stateChanged(self, state):
        if state:
            self.ui_interfacewin.checkBox_PCA_RandomState.setEnabled(False)
        else:
            self.ui_interfacewin.checkBox_PCA_RandomState.setEnabled(True)
    @pyqtSlot()
    def on_PCA_Btn_clicked(self):
        count = self.ui_interfacewin.listWidget_PCA_2.count()
        self.ui_interfacewin.spinBoxn_PCA_n_components_int.setMaximum(count)
        field = []
        for i in range(0, count):
            fieldname = self.ui_interfacewin.listWidget_PCA_2.item(i).text()
            field.append(fieldname)
        fieldvalues = self.copydf[field]
        n_components = None
        copy = False
        whiten = False
        svd_solver = 'auto'
        tol = 0.0
        iterated_power = 'auto'
        random_state = None
        #n_components
        if self.ui_interfacewin.checkBox_PCA_n_components_double.isChecked():
            n_components = self.ui_interfacewin.doubleSpinBox_PCA_n_components_double.value()
        if self.ui_interfacewin.checkBox_PCA_n_components_mle.isChecked():
            n_components = 'mle'
        if self.ui_interfacewin.checkBox_PCA_n_components_int.isChecked():
            n_components = self.ui_interfacewin.spinBoxn_PCA_n_components_int.value()
        #svd_solver
        if self.ui_interfacewin.radioButton_PCA_auto_solver_auto.isChecked():
            svd_solver = 'auto'
        if self.ui_interfacewin.radioButton__PCA_auto_solver_full.isChecked():
            svd_solver = 'full'
        if self.ui_interfacewin.radioButton__PCA_auto_solver_arpack.isChecked():
            svd_solver = 'arpack'
        if self.ui_interfacewin.radioButton__PCA_auto_solver_randomized.isChecked():
            svd_solver = 'randomized'
        #random_state
        if self.ui_interfacewin.checkBox_PCA_RandomState.isChecked():
            random_state = 'RandomState'
        if self.ui_interfacewin.checkBox_PCA_random_state_int.isChecked():
            random_state = self.ui_interfacewin.spinBox_PCA_random_state_int.value()
        #tol
        tol = self.ui_interfacewin.doubleSpinBox_PCA_tol.value()
        #whiten
        if self.ui_interfacewin.checkBox_PCA_whiten.isChecked():
            whiten = True
        #copy
        if self.ui_interfacewin.checkBox_PCA_copy.isChecked():
            copy = True
        #iterated_power
        if self.ui_interfacewin.checkBox_PCA_iterated_power.isChecked():
            iterated_power = self.ui_interfacewin.spinBox_PCA_iterated_power.value()

        lower_dim = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)
        lower_dim.fit(fieldvalues.values)
        self.ui_interfacewin.textEdit_result.append("%s列主成分分析(PCA降维)："%(field))
        self.ui_interfacewin.textEdit_result.append("保留的成分个数：%s" % (lower_dim.n_components_))
        self.ui_interfacewin.textEdit_result.append("具有最大方差的成分：%s"%(lower_dim.components_))
        self.ui_interfacewin.textEdit_result.append("转化后的数值：%s"%(lower_dim.explained_variance_))
        self.ui_interfacewin.textEdit_result.append("纬度的重要性：%s" % (lower_dim.explained_variance_ratio_))
        self.ui_interfacewin.textEdit_result.append("每个特征的奇异值：%s" % (lower_dim.singular_values_))
        self.ui_interfacewin.textEdit_result.append("每个特征均值：%s" % (lower_dim.mean_))
        self.ui_interfacewin.textEdit_result.append("估计噪声的协方差：%s" % (lower_dim.noise_variance_))

    #离散属性相关性分析
    #添加项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_discrete_correlation_itemClicked(self, item):
        itemname = item.text()
        self.ui_interfacewin.listWidget_discrete_correlation_2.addItem(itemname)
    #删除项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_discrete_correlation_2_itemClicked(self, item):
        currow = self.ui_interfacewin.listWidget_discrete_correlation_2.currentIndex().row()
        self.ui_interfacewin.listWidget_discrete_correlation_2.takeItem(currow)
    #检测
    @pyqtSlot()
    def on_discrete_correlation_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_discrete_correlation_2.count()
        if fieldcount > 0:
            for i in range(0, fieldcount):
                field = self.ui_interfacewin.listWidget_discrete_correlation_2.item(i).text()
                fieldvalues = self.copydf[field]
                if self.ui_interfacewin.checkBox_discrete_correlation_2.isChecked():
                    result = self.getProbSS(fieldvalues)
                    self.ui_interfacewin.textEdit_result.append("%s列可能性平方和:%s" % (field, result))
                if self.ui_interfacewin.checkBox_discrete_correlation_3.isChecked():
                    result = self.getEntropy(fieldvalues)
                    self.ui_interfacewin.textEdit_result.append("%s列熵:%s" % (field, result))
            if fieldcount > 1:
                fieldcount_2 = fieldcount - 1
                for i in range(0, fieldcount_2):
                    j = i + 1
                    field_1 = self.ui_interfacewin.listWidget_discrete_correlation_2.item(i).text()
                    field_2 = self.ui_interfacewin.listWidget_discrete_correlation_2.item(j).text()
                    fieldvalues_1 = self.copydf[field_1]
                    fieldvalues_2 = self.copydf[field_2]
                    if self.ui_interfacewin.checkBox_discrete_correlation.isChecked():
                        result = self.getGini(fieldvalues_1=fieldvalues_1, fieldvalues_2=fieldvalues_2)
                        self.ui_interfacewin.textEdit_result.append("%s和%s列Gini:%s" % (field_1, field_2, result))
                    if self.ui_interfacewin.checkBox_discrete_correlation_4.isChecked():
                        result = self.getCondEntropy(fieldvalues_1, fieldvalues_2)
                        self.ui_interfacewin.textEdit_result.append("%s和%s列条件熵:%s" % (field_1, field_2, result))
                    if self.ui_interfacewin.checkBox_discrete_correlation_5.isChecked():
                        result = self.getEntropyGain(fieldvalues_1, fieldvalues_2)
                        self.ui_interfacewin.textEdit_result.append("%s和%s列熵增益:%s" % (field_1, field_2, result))
                    if self.ui_interfacewin.checkBox_discrete_correlation_6.isChecked():
                        result = self.getEntropyGainRatio(fieldvalues_1, fieldvalues_2)
                        self.ui_interfacewin.textEdit_result.append("%s和%s列熵增益率:%s" % (field_1, field_2, result))
                    if self.ui_interfacewin.checkBox_discrete_correlation_7.isChecked():
                        result = self.getDiscreteRelation(fieldvalues_1, fieldvalues_2)
                        self.ui_interfacewin.textEdit_result.append("%s和%s列相关度:%s" % (field_1, field_2, result))
    #LDA降维
    #添加条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_LDA_itemClicked(self, item):
        itemname = item.text()
        if self.ui_interfacewin.checkBox_LDA_add_labels.isChecked():
            self.ui_interfacewin.listWidget_LDA_3.clear()
            self.ui_interfacewin.listWidget_LDA_3.addItem(itemname)
        else:
            self.ui_interfacewin.listWidget_LDA_2.addItem(itemname)
    #删除条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_LDA_2_itemClicked(self, item):
        currow = self.ui_interfacewin.listWidget_LDA_2.currentIndex().row()
        self.ui_interfacewin.listWidget_LDA_2.takeItem(currow)
    #降维
    @pyqtSlot()
    def on_LDA_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_LDA_2.count()
        fieldcount_2 = self.ui_interfacewin.listWidget_LDA_3.count()
        if fieldcount > 1 and fieldcount_2 > 0:
            store_covariance = False
            n_components = None
            if self.ui_interfacewin.radioButton_LDA_solver_svd.isChecked():
                solver = 'svd'
            if self.ui_interfacewin.radioButton_LDA_solver_lsqr.isChecked():
                solver = 'lsqr'
            if self.ui_interfacewin.radioButton_LDA_solver_eigen.isChecked():
                solver = 'eigen'
            if self.ui_interfacewin.radioButton_LDA_shrinkage_None.isChecked():
                shrinkage = None
            if self.ui_interfacewin.radioButton_LDA_shrinkage_auto.isChecked():
                shrinkage = 'auto'
            if self.ui_interfacewin.radioButton_LDA_shrinkage_float.isChecked():
                shrinkage = self.ui_interfacewin.doubleSpinBox_LDA_float.value()
            if self.ui_interfacewin.checkBox_LDA_store_covariance.isChecked():
                store_covariance = True
            if self.ui_interfacewin.checkBox_LDA_n_components.isChecked():
                n_components = self.ui_interfacewin.spinBox_LDA_n_components.value()
            tol = self.ui_interfacewin.doubleSpinBox_LDA_tol.value()

            fields = []
            for i in range(0, fieldcount):
                field = self.ui_interfacewin.listWidget_LDA_2.item(i).text()
                fields.append(field)
            field_label = self.ui_interfacewin.listWidget_LDA_3.item(0).text()
            label_values = self.copydf[field_label]
            fieldvalues = self.copydf[fields]

            clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, n_components=n_components,
                                             store_covariance=store_covariance, tol=tol)
            clf.fit(fieldvalues, label_values)
            self.ui_interfacewin.textEdit_result.append("%s列LDA降维:" % (fields))
            self.ui_interfacewin.textEdit_result.append("coef_:%s" % (clf.coef_))
            self.ui_interfacewin.textEdit_result.append("intercept_:%s" % (clf.intercept_))
            #self.ui_interfacewin.textEdit_result.append("covariance_:%s" % (clf.covariance_))
            self.ui_interfacewin.textEdit_result.append("explained_variance_ratio_:%s" % (clf.explained_variance_ratio_))
            self.ui_interfacewin.textEdit_result.append("means_:%s" % (clf.means_))
            self.ui_interfacewin.textEdit_result.append("priors_:%s" % (clf.priors_))
            self.ui_interfacewin.textEdit_result.append("scalings_:%s" % (clf.scalings_))
            self.ui_interfacewin.textEdit_result.append("xbar_:%s" % (clf.xbar_))
            self.ui_interfacewin.textEdit_result.append("classes_:%s" % (clf.classes_))

    #特征工程
    #条件框变化
    @pyqtSlot(int)
    def on_comboBox_data_cleaning_2_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_feature.setCurrentIndex(0)
        if index == 1:
            self.ui_interfacewin.stackedWidget_feature.setCurrentIndex(1)
        if index == 2:
            self.ui_interfacewin.stackedWidget_feature.setCurrentIndex(2)
    #采样条件框变化
    @pyqtSlot(int)
    def on_comboBox_sample_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_sample.setCurrentIndex(0)
        if index == 1:
            self.ui_interfacewin.stackedWidget_sample.setCurrentIndex(1)
    #取样本
    #添加col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_sample_itemClicked(self, item):
        itemtext = item.text()
        if self.ui_interfacewin.checkBox_sample_2.isChecked():
            self.ui_interfacewin.listWidget_sample_2.clear()
            field = self.copydf[itemtext].unique()
            fieldlist = field.tolist()
            new_fieldlist = [str(x) for x in fieldlist]
            self.ui_interfacewin.listWidget_sample_2.addItems(new_fieldlist)
        else:
            self.ui_interfacewin.listWidget_sample_2.addItem(itemtext)
    #删除col
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_sample_2_itemClicked(self, item):
        if not self.ui_interfacewin.checkBox_sample_2.isChecked():
            currow = self.ui_interfacewin.listWidget_sample_2.currentIndex().row()
            self.ui_interfacewin.listWidget_sample_2.takeItem(currow)
   #删除条目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_sample_2_itemDoubleClicked(self, item):
        if self.ui_interfacewin.checkBox_sample_2.isChecked():
            currow = self.ui_interfacewin.listWidget_sample_2.currentIndex().row()
            self.ui_interfacewin.listWidget_sample_2.takeItem(currow)
    #选择整体数据条件框变化
    @pyqtSlot(int)
    def on_checkBox_sample_stateChanged(self, state):
        if state:
            self.ui_interfacewin.listWidget_sample.setEnabled(False)
            self.ui_interfacewin.listWidget_sample_2.setEnabled(False)
        else:
            self.ui_interfacewin.listWidget_sample.setEnabled(True)
            self.ui_interfacewin.listWidget_sample_2.setEnabled(True)

    #上采样或过采样条件框变化
    @pyqtSlot(int)
    def on_checkBox_sample_2_stateChanged(self, state):
        if state:
            self.ui_interfacewin.checkBox_sample.setEnabled(False)
            self.ui_interfacewin.sample_Btn.setEnabled(False)
        else:
            self.ui_interfacewin.checkBox_sample.setEnabled(True)
            self.ui_interfacewin.sample_Btn.setEnabled(True)
     #普通采样
    @pyqtSlot()
    def on_sample_Btn_clicked(self):
        n = None
        frac = None
        replace = False
        weights = None
        random_state = None
        axis = None
        if self.ui_interfacewin.radioButton_sample_n.isChecked():
            n = self.ui_interfacewin.spinBox_sample_n.value()
        if self.ui_interfacewin.radioButton_sample_frac.isChecked():
            frac = self.ui_interfacewin.doubleSpinBox_sample_frac.value()
        if self.ui_interfacewin.checkBox_sample_replace.isChecked():
            replace = True
        if self.ui_interfacewin.checkBox_sample_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_sample_random_state.value()
        if self.ui_interfacewin.checkBox_sample_axis.isChecked():
            axis = 1
        if self.ui_interfacewin.checkBox_sample.isChecked():
            sampleresult = self.copydf.sample(n=n, frac=frac, replace=replace, weights=None, random_state=random_state, axis=axis)
            if self.ui_interfacewin.checkBox_sample_random_state.isChecked():
                self.ui_interfacewin.textEdit_result.append("数据集抽样结果(随机数为%s)：\n%s" % (random_state, sampleresult))
            else:
                self.ui_interfacewin.textEdit_result.append("数据集抽样结果(未设置随机数)：\n%s" % (sampleresult))
        else:
            fieldcount = self.ui_interfacewin.listWidget_sample_2.count()
            fields = []
            for i in range(0, fieldcount):
                field = self.ui_interfacewin.listWidget_sample.item(i).text()
                fields.append(field)
            fieldvalues = self.copydf[fields]
            sampleresult = fieldvalues.sample(n=n, frac=frac, replace=replace, weights=None, random_state=random_state, axis=axis)
            if self.ui_interfacewin.checkBox_sample_random_state.isChecked():
                self.ui_interfacewin.textEdit_result.append("%s列抽样结果(随机数为%s)：\n%s" % (fields, random_state, sampleresult))
            else:
                self.ui_interfacewin.textEdit_result.append("%s列抽样结果(未设置随机数)：\n%s" % (fields, sampleresult))
        ok = QMessageBox.critical(self, '提示', "是否保存样本", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            savefile = SaveFile()
            savefile.open()
            if savefile.exec() == QDialog.Accepted:
                sep, na_rep, encoding, compression, decimal, savefilename, index, header = savefile.save()
                sampleresult.to_csv(path_or_buf=savefilename, sep=sep, na_rep=na_rep, header=header, index=index,
                                   encoding=encoding, compression=compression, decimal=decimal)
                self.ui_interfacewin.textEdit_result.append("样本已保存：%s" % (savefilename))
    #显示概况
    @pyqtSlot()
    def on_survey_Btn_clicked(self):
        field = self.ui_interfacewin.listWidget_sample.currentItem().text()
        labels = self.copydf[field]
        fieldcount = self.ui_interfacewin.listWidget_sample_2.count()
        for i in range(0, fieldcount):
            fieldvalues = self.ui_interfacewin.listWidget_sample_2.item(i).text()
            if is_number(fieldvalues):
                fieldvalues = float(fieldvalues)
            self.ui_interfacewin.textEdit_result.append("%s列%s:%s" % (field, fieldvalues, len(labels[labels == fieldvalues])))

    #下采样
    @pyqtSlot()
    def on_downsampling_Btn_clicked(self):
        curcol = self.ui_interfacewin.listWidget_sample.currentItem().text()
        X = self.copydf.iloc[:, self.copydf.columns != curcol]
        y = self.copydf.iloc[:, self.copydf.columns == curcol]
        curcolvalues = self.ui_interfacewin.listWidget_sample_2.currentItem().text()
        if is_number(curcolvalues):
            curcolvalues = float(curcolvalues)
        fieldvalues = self.copydf[curcol]
        number_records_fraud = len(self.copydf[fieldvalues == curcolvalues])
        fraud_indices = np.array(self.copydf[fieldvalues == curcolvalues].index)
        normal_indices = self.copydf[fieldvalues != curcolvalues].index
        try:
            random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)#随机选择,在normal_indices中选择
        except:
            QMessageBox.critical(self, 'Error', "选取的样本数量是多的", QMessageBox.Ok)
            return
        random_normal_indices = np.array(random_normal_indices)
        under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])#连接
        under_sample_data = self.copydf.iloc[under_sample_indices, :]#定位
        X_undersample = under_sample_data.iloc[:, under_sample_data.columns != curcol]
        y_undersample = under_sample_data.iloc[:, under_sample_data.columns == curcol]
        self.ui_interfacewin.textEdit_result.append("%s列%s样本比例:%s" % (curcol, curcolvalues,
                                                                      len(under_sample_data[under_sample_data[curcol]==curcolvalues]) / len(under_sample_data)))
        self.ui_interfacewin.textEdit_result.append("%s列非%s样本比例:%s" % (curcol, curcolvalues,
                                                                       len(under_sample_data[under_sample_data[curcol]!=curcolvalues]) / len(under_sample_data)))
        self.ui_interfacewin.textEdit_result.append("依据%s列%s下采样样本长度：%s" % (curcol, curcolvalues, len(under_sample_data)))
        ok = QMessageBox.critical(self, '提示', "是否保存样本", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            savefile = SaveFile()
            savefile.open()
            if savefile.exec() == QDialog.Accepted:
                sep, na_rep, encoding, compression, decimal, savefilename, index, header = savefile.save()
                under_sample_data.to_csv(path_or_buf=savefilename, sep=sep, na_rep=na_rep, header=header, index=index,
                                   encoding=encoding, compression=compression, decimal=decimal)
                self.ui_interfacewin.textEdit_result.append("样本已保存：%s" % (savefilename))

    #过采样
    @pyqtSlot()
    def on_oversampling_Btn_clicked(self):
        columns = self.copydf.columns
        features_columns = columns.delete(len(columns) - 1)
        features = self.copydf[features_columns]
        field = self.ui_interfacewin.listWidget_sample.currentItem().text()
        labels = self.copydf[field]
        fieldvalues = self.ui_interfacewin.listWidget_sample_2.currentItem().text()

        if is_number(fieldvalues):
            fieldvalues = float(fieldvalues)
        #features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=0)
        oversampler = SMOTE(random_state=0)
        try:
            os_features, os_labels = oversampler.fit_sample(features, labels)#不能有字符
        except:
            QMessageBox.critical(self, 'Error', "数据中含有字符列,请做处理或者删除(不建议删除)", QMessageBox.Ok)
            return
        self.ui_interfacewin.textEdit_result.append("下取样正负样本长度:%s" % (len(os_labels[os_labels == fieldvalues])))

        ok = QMessageBox.critical(self, '提示', "是否保存样本", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            savefile = SaveFile()
            savefile.open()
            if savefile.exec() == QDialog.Accepted:
                sep, na_rep, encoding, compression, decimal, savefilename, index, header = savefile.save()
                os_labels.to_csv(path_or_buf=savefilename, sep=sep, na_rep=na_rep, header=header, index=index,
                                    encoding=encoding, compression=compression, decimal=decimal)
                self.ui_interfacewin.textEdit_result.append("样本已保存：%s" % (savefilename))

    #特征选择
    #条件框变化
    @pyqtSlot(int)
    def on_comboBox_features_selection_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_features_selection.setCurrentIndex(0)
        if index == 1:
            self.ui_interfacewin.stackedWidget_features_selection.setCurrentIndex(1)
        if index == 2:
            self.ui_interfacewin.stackedWidget_features_selection.setCurrentIndex(2)
    #添加项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_features_selection_itemClicked(self, item):
        itemname = item.text()
        if self.ui_interfacewin.checkBox_features_selection.isChecked():
            self.ui_interfacewin.listWidget_features_selection_3.clear()
            self.ui_interfacewin.listWidget_features_selection_3.addItem(itemname)
        else:
            self.ui_interfacewin.listWidget_features_selection_2.addItem(itemname)

    #删除项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_features_selection_2_itemClicked(self, item):
        currow = self.ui_interfacewin.listWidget_features_selection_2.currentIndex().row()
        self.ui_interfacewin.listWidget_features_selection_2.takeItem(currow)

    #过滤
    @pyqtSlot()
    def on_features_selection_filter_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_selection_2.count()
        labelcount = self.ui_interfacewin.listWidget_features_selection_3.count()
        if fieldcount > 1 and labelcount > 0:
            fields = []
            for i in range(0, fieldcount):
                field = self.ui_interfacewin.listWidget_features_selection_2.item(i).text()
                fields.append(field)
            label = self.ui_interfacewin.listWidget_features_selection_3.item(0).text()
            labelvalues = self.copydf[label]
            fieldvalues = self.copydf[fields]
            if self.ui_interfacewin.radioButton_features_selection_filter_all.isChecked():
                k = "all"
            if self.ui_interfacewin.radioButton_features_selection_filter_int.isChecked():
                k = self.ui_interfacewin.spinBox_features_selection_filter_int.value()
            skb = SelectKBest(k=k)
            skb.fit(fieldvalues, labelvalues)
            self.ui_interfacewin.textEdit_result.append("%s列,标注%s列过滤(过滤思想):%s" %
                                                        (fields, label, skb.transform(fieldvalues)))

    #包裹
    @pyqtSlot()
    def on_features_selection_package_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_selection_2.count()
        labelcount = self.ui_interfacewin.listWidget_features_selection_3.count()
        if fieldcount > 1 and labelcount > 0:
            fields = []
            for i in range(0, fieldcount):
                field = self.ui_interfacewin.listWidget_features_selection_2.item(i).text()
                fields.append(field)
            label = self.ui_interfacewin.listWidget_features_selection_3.item(0).text()
            labelvalues = self.copydf[label]
            fieldvalues = self.copydf[fields]
            n_features_to_select = None
            if self.ui_interfacewin.checkBox_features_selection_package_n_features_to_select.isChecked():
                n_features_to_select = self.ui_interfacewin.spinBox_features_selection_package_n_features_to_select.value()
            step = self.ui_interfacewin.doubleSpinBox_features_selection_package_step.value()
            verbose = self.ui_interfacewin.spinBox_features_selection_package_verbose.value()

            rfe = RFE(estimator=SVR(kernel="linear"),
                      n_features_to_select=n_features_to_select, step=100)
            self.ui_interfacewin.textEdit_result.append("%s列标注%s过滤(包裹思想):%s" %
                                                        (fields, label, rfe.fit_transform(fieldvalues, labelvalues)))

    #嵌入
    @pyqtSlot()
    def on_features_selection_nest_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_selection_2.count()
        labelcount = self.ui_interfacewin.listWidget_features_selection_3.count()
        if fieldcount > 1 and labelcount > 0:
            fields = []
            for i in range(0, fieldcount):
                field = self.ui_interfacewin.listWidget_features_selection_2.item(i).text()
                fields.append(field)
            label = self.ui_interfacewin.listWidget_features_selection_3.item(0).text()
            labelvalues = self.copydf[label]
            fieldvalues = self.copydf[fields]
            threshold = None
            prefit = False
            max_features = None
            if self.ui_interfacewin.checkBox_features_selection_nest_threshold.isChecked():
                scaling_factor = self.ui_interfacewin.doubleSpinBox_features_selection_nest_threshold.value()
                threshold =self.ui_interfacewin.comboBox_features_selection_nest_threshold.currentText()
            if self.ui_interfacewin.checkBox_features_selection_nest_prefit.isChecked():
                prefit = True
            norm_order = self.ui_interfacewin.spinBox_features_selection_nest_norm_order.value()
            if self.ui_interfacewin.checkBox_features_selection_nest_max_features.isChecked():
                max_features = self.ui_interfacewin.spinBox_features_selection_nest_max_features.value()
            sfm = SelectFromModel(estimator=DecisionTreeRegressor(), threshold=threshold, prefit=prefit,
                                  norm_order=norm_order, max_features=max_features)
            self.ui_interfacewin.textEdit_result.append("%s列标注%s过滤(嵌入思想):%s" %
                                                        (fields, label, sfm.fit_transform(fieldvalues, labelvalues)))
    #特征变换
    #添加项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_features_transform_itemClicked(self, item):
        itemname = item.text()
        self.ui_interfacewin.listWidget_features_transform_2.addItem(itemname)

    #删除项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_features_transform_2_itemClicked(self, item):
        currow = self.ui_interfacewin.listWidget_features_transform_2.currentIndex().row()
        self.ui_interfacewin.listWidget_features_transform_2.takeItem(currow)

    #条件框变化
    @pyqtSlot(int)
    def on_comboBox_features_transform_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_features_transform.setCurrentIndex(0)
        if index == 1:
            self.ui_interfacewin.stackedWidget_features_transform.setCurrentIndex(1)
        if index == 2:
            self.ui_interfacewin.stackedWidget_features_transform.setCurrentIndex(2)
        if index == 3:
            self.ui_interfacewin.stackedWidget_features_transform.setCurrentIndex(3)
        if index == 4:
            self.ui_interfacewin.stackedWidget_features_transform.setCurrentIndex(4)
        if index == 5:
            self.ui_interfacewin.stackedWidget_features_transform.setCurrentIndex(5)
        if index == 6:
            self.ui_interfacewin.stackedWidget_features_transform.setCurrentIndex(6)

    #获取操作列
    def features_transform_get_opeator_col(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            if fieldcount == 1:
                field = self.ui_interfacewin.listWidget_features_transform_2.item(0).text()
                fieldvalues = np.array(self.copydf[field]).reshape(1, -1)
                return field, fieldvalues
            else:
                fields = []
                for i in range(0, fieldcount):
                    field = self.ui_interfacewin.listWidget_features_transform_2.item(i).text()
                    fields.append(field)
                fieldvalues = np.array(self.copydf[fields]).reshape(-1, 1)
                return fields, fieldvalues

    #对数化
    @pyqtSlot()
    def on_features_transform_log_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            field, fieldvalues = self.features_transform_get_opeator_col()
            result = np.log(fieldvalues)
            self.copydf[field] = result
            self.ui_interfacewin.textEdit_result.append(("%s列对数化:%s" % (field, result)))

    #指数化
    @pyqtSlot()
    def on_features_transform_indexation_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            field, fieldvalues = self.features_transform_get_opeator_col()
            result = np.exp(fieldvalues)
            self.copydf[field] = result
            self.ui_interfacewin.textEdit_result.append(("%s列指数化:%s" % (field, result)))

    #归一化
    @pyqtSlot()
    def on_features_transform_normalization_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            if fieldcount == 1:
                field = self.ui_interfacewin.listWidget_features_transform_2.item(0).text()
                fieldvalues = np.array(self.copydf[field]).reshape(-1, 1)
                #fieldvalues = LabelEncoder().fit_transform(field).reshape(-1, 1)
                result = MinMaxScaler().fit_transform(fieldvalues)
                self.copydf[field] = result
                self.ui_interfacewin.textEdit_result.append(
                    ("%s列归一化:%s" % (field, MinMaxScaler().fit_transform(fieldvalues))))
            else:
                fields = []
                for i in range(0, fieldcount):
                    field = self.ui_interfacewin.listWidget_features_transform_2.item(i).text()
                    fields.append(field)
                fieldvalues = np.array(self.copydf[fields]).reshape(-1, 1)

                self.copydf[fields] = MinMaxScaler().fit_transform(fieldvalues)
                self.ui_interfacewin.textEdit_result.append(("%s列归一化:%s" % (fields, MinMaxScaler().fit_transform(fieldvalues))))

    #标准化
    @pyqtSlot()
    def on_features_transform_standardization_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            if fieldcount == 1:
                field = self.ui_interfacewin.listWidget_features_transform_2.item(0).text()
                fieldvalues = np.array(self.copydf[field]).reshape(-1, 1)
                result = StandardScaler().fit_transform(fieldvalues)
                self.copydf[field] = result
                self.ui_interfacewin.textEdit_result.append(("%s列标准化:%s" % (field, result)))
            else:
                fields = []
                for i in range(0, fieldcount):
                    field = self.ui_interfacewin.listWidget_features_transform_2.item(i).text()
                    fields.append(field)
                fieldvalues = np.array(self.copydf[fields]).reshape(-1, 1)
                result = StandardScaler().fit_transform(fieldvalues)
                self.copydf[field] = result
                self.ui_interfacewin.textEdit_result.append(("%s列标准化:%s" % (field, result)))
    #标签化
    @pyqtSlot()
    def on_features_transform_label_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            if fieldcount == 1:
                field = self.ui_interfacewin.listWidget_features_transform_2.item(0).text()
                fieldvalues = np.array(self.copydf[field])
                result = LabelEncoder().fit_transform(fieldvalues)
                self.copydf[field] = result
                self.ui_interfacewin.textEdit_result.append(("%s列标签化:%s" % (field, result)))
            else:
                for i in range(0, fieldcount):
                    field = self.ui_interfacewin.listWidget_features_transform_2.item(i).text()
                    fieldvalues = np.array(self.copydf[field])
                    result = LabelEncoder().fit_transform(fieldvalues)
                    self.copydf = pd.get_dummies(self.copydf, columns=[field])
                    self.ui_interfacewin.textEdit_result.append(("%s列标签化:%s" % (field, result)))

    #独热编码
    @pyqtSlot()
    def on_features_transform_one_hot_Btn_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            if fieldcount == 1:
                field = self.ui_interfacewin.listWidget_features_transform_2.item(0).text()
                fieldvalues = self.copydf[field]
                #fieldunique = np.array(self.copydf[field].unique())
                lb_encoder = LabelEncoder()  # 独热编码，开始
                lb_encoder = lb_encoder.fit(np.array(fieldvalues))
                lb_trans_f = lb_encoder.transform(np.array(fieldvalues))
                oht_enoder = OneHotEncoder().fit(lb_trans_f.reshape(-1, 1))
                self.copydf[field] = lb_trans_f
                result = oht_enoder.transform(lb_trans_f.reshape(-1, 1))
                result2 = result.toarray()
                #self.copydf = pd.get_dummies(self.copydf, columns=[field])
                self.copydf = pd.get_dummies(self.copydf, columns=[field])
                self.ui_interfacewin.textEdit_result.append(("%s列独热编码:%s" % (field, result2)))

            else:
                for i in range(0, fieldcount):
                    fieldname = self.ui_interfacewin.listWidget_features_transform_2.item(i).text()
                    fieldvalues = self.copydf[fieldname]
                    lb_encoder = LabelEncoder()  # 独热编码，开始
                    lb_encoder = lb_encoder.fit(np.array(fieldvalues))
                    lb_trans_f = lb_encoder.transform(np.array(fieldvalues))
                    oht_enoder = OneHotEncoder().fit(lb_trans_f.reshape(-1, 1))
                    result = oht_enoder.transform(lb_encoder.transform(np.array(fieldvalues)).reshape(-1, 1)).toarray()
                    self.copydf[fieldname] = result
                    self.ui_interfacewin.textEdit_result.append(("%s列独热编码:%s" % (fieldname, result)))

            self.standardmodel.clear()
            rownum, colnum = self.copydf.shape
            col_names = self.copydf.columns.tolist()
            if rownum < 1000:
                for col in range(colnum):
                    colname = []
                    rows = self.copydf[col_names[col]].tolist()
                    for row in range(rownum):
                        item = QStandardItem("%s" % (rows[row]))
                        colname.append(item)
                    self.standardmodel.insertColumn(col, colname)
            if rownum > 1000:
                for col in range(colnum):
                    colname = []
                    rows = self.copydf[col_names[col]].tolist()
                    for row in range(1, 1000):
                        item = QStandardItem("%s" % (rows[row]))
                        colname.append(item)
                    self.standardmodel.insertColumn(col, colname)
            self.standardmodel.setHorizontalHeaderLabels(col_names)
    #规范化
    @pyqtSlot()
    def on_features_transform_normalization_Btn_2_clicked(self):
        fieldcount = self.ui_interfacewin.listWidget_features_transform_2.count()
        if fieldcount > 0:
            field, fieldvalues = self.features_transform_get_opeator_col()
            result = Normalizer(norm='l1').fit_transform(fieldvalues)
            self.copydf[field] = result
            self.ui_interfacewin.textEdit_result.append("%s列规范化:%s" % (field, result))

        #self.plotly_show.show_line_plots(x, y)
        #self.plotly_show.show_bar_charts(x, y)
    #拉格朗日插值法
    def ployinter(self, s, n, k=5):
        # s为列向量，n为被插值位置，k为取前后的数据个数
        y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
        y = y[y.notnull()]  # 剔除空值
        return lagrange(y.index, list(y))(n)

    #Gini
    def getGini(self, fieldvalues_1, fieldvalues_2):
        assert(len(fieldvalues_1) == len(fieldvalues_2))
        d = dict()
        for i in list(range(len(fieldvalues_1))):
            d[fieldvalues_1[i]] = d.get(fieldvalues_1[i], []) + [fieldvalues_2[i]]
        return 1 - sum([self.getProbSS(d[k]) * len(d[k]) / float(len(fieldvalues_1)) for k in d])

    #可能性平方和
    def getProbSS(self, fieldvalues):
        if not isinstance(fieldvalues, pd.core.series.Series):
            fieldvalues = pd.Series(fieldvalues)
        prt_ary = np.array(pd.groupby(fieldvalues, by=fieldvalues).count().values / float(len(fieldvalues)))
        return sum(prt_ary ** 2)

    #熵
    def getEntropy(self, fieldvalues):
        if not isinstance(fieldvalues, pd.core.series.Series):
            fieldvalues = pd.Series(fieldvalues)
        prt_ary = np.array(pd.groupby(fieldvalues, by=fieldvalues).count().values / float(len(fieldvalues)))
        return -(np.log2(prt_ary) * prt_ary).sum()

    #条件熵
    def getCondEntropy(self, fieldvalues_1, fieldvalues_2):
        assert (len(fieldvalues_1) == len(fieldvalues_2))
        d = dict()
        for i in list(range(len(fieldvalues_1))):
            d[fieldvalues_1[i]] = d.get(fieldvalues_1[i], []) + [fieldvalues_2[i]]
        return sum([self.getEntropy(d[k]) * len(d[k]) / float(len(fieldvalues_1)) for k in d])

    #熵增益
    def getEntropyGain(self, fieldvalues_1, fieldvalues_2):
        return self.getEntropy(fieldvalues_2) - self.getCondEntropy(fieldvalues_1, fieldvalues_2)

    #熵增益率
    def getEntropyGainRatio(self, fieldvalues_1, fieldvalues_2):
        return self.getEntropyGain(fieldvalues_1, fieldvalues_2) / self.getEntropy(fieldvalues_2)

    #相关度
    def getDiscreteRelation(self, fieldvalues_1, fieldvalues_2):
        return self.getEntropyGain(fieldvalues_1, fieldvalues_2) / math.sqrt(self.getEntropy(fieldvalues_1) * self.getEntropy(fieldvalues_2))

    #################################################################建模模块##########################################################
    @pyqtSlot(int)
    #建模类型变化
    def on_comboBox_model_currentIndexChanged(self, index):
        self.ui_interfacewin.stackedWidget_2.setCurrentIndex(index)
    #关联分析条件变化
    @pyqtSlot(int)
    def on_comboBox_correlation_analysis_Apriori_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(16)
    #半监督条件框变化
    @pyqtSlot(int)
    def on_comboBox_semi_supervised_learning_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(15)
    #聚类条件框变化
    @pyqtSlot(int)
    def on_comboBox_cluster_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(17)
        if index == 1:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(18)
        if index == 2:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(19)

    #分类回归条件框变化
    @pyqtSlot(int)
    def on_comboBox_model_classifier_type_currentIndexChanged(self, index):
        if index == 0:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 1:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 2:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 3:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 4:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 5:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 6:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 7:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 8:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 9:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 10:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 11:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 12:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 13:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
        if index == 14:
            self.ui_interfacewin.stackedWidget_model_classifier.setCurrentIndex(index)
    #分类与回归
    #Dense
    @pyqtSlot()
    def on_model_classifier_Dense_clicked(self):
        features, label, label_count = self.split_data2()
        self.hr_modeling_nn(features, label, label_count)
    def hr_modeling_nn(self, features, label, label_count):
        #f_v = self.copydf.values
        f_v = features.values
        f_names = features.columns.values
        l_v = label
        X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
        #X_train, X_test, Y_train, Y_test, X_validation, Y_validation = self.cut_data()
        optimizer = self.ui_interfacewin.comboBox_classifier_Dense_optimizer.currentText()
        lr = self.ui_interfacewin.doubleSpinBox_classifier_Dense_lr.value()
        loss = self.ui_interfacewin.comboBox_classifier_Dense_loss.currentText()
        Dense_count = self.ui_interfacewin.spinBoxclassifier_Dense_next_number.value()
        activation2 = self.ui_interfacewin.comboBox_classifier_Dense_Activation_2.currentText()
        activation = self.ui_interfacewin.comboBox_classifier_Dense_Activation.currentText()
        nb_epoch = self.ui_interfacewin.spinBox_classifier_Dense_iterations.value()
        batch_size = self.ui_interfacewin.spinBox_classifier_Dense_update_count.value()
        mdl = Sequential()
        mdl.add(Dense(Dense_count, input_dim=len(f_v[0])))
        mdl.add(Activation(activation))
        mdl.add(Dense(label_count))
        mdl.add(Activation(activation2))
        sgd = SGD(lr=lr)
        mdl.compile(loss=loss, optimizer=optimizer, metrics=None, loss_weights=None,
                    sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
        mdl.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size)
        ok = QMessageBox.question(self, "提示", "是否保存模型?", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            joblib.dump(mdl, 'model/Sequential_model')
        #np.array([[0, 1] if i == 1 else [1, 0] for i in Y_train])
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = mdl.predict(X_part)
            self.ui_interfacewin.textEdit_result.append("%s" % str(Y_pred))
            len_Y_pred = len(Y_pred - 1)
            Y_pred = Y_part
            #Y_pred = np.array(Y_pred[:, 1].reshape(1, -1))[0]
            #self.ui_interfacewin.textEdit_result.append("ACC:%s" % (accuracy_score(Y_part, Y_pred)))
            #self.ui_interfacewin.textEdit_result.append("REC:%s" % (recall_score(Y_part, Y_pred)))
            #self.ui_interfacewin.textEdit_result.append("F1:%s" % (f1_score(Y_part, Y_pred)))
            fpr, tpr, threshold = roc_curve(Y_part, Y_pred)
            self.ui_interfacewin.textEdit_result.append("AUC:%s" % (auc(fpr, tpr)))
            self.ui_interfacewin.textEdit_result.append("AUC_Score:%s" % (roc_auc_score(Y_part, Y_pred)))

    # 切割数据集
    def split_data(self):
        if self.is_split_data == False and self.data_label_cur_number == None:
            self.copydf2 = self.copydf.copy()
            field = self.ui_interfacewin.comboBox_model_label.currentText()
            self.label = self.copydf2[field]

            self.copydf2 = self.copydf2.drop(field, axis=1)
            features = self.copydf2
            self.is_split_data = True
            self.data_label_cur_number = self.ui_interfacewin.comboBox_model_label.currentIndex()
            return features, self.label
        if self.is_split_data and self.data_label_cur_number != None:
            curindex = self.ui_interfacewin.comboBox_model_label.currentIndex()
            if self.data_label_cur_number == curindex:
                features = self.copydf2
                self.label = self.label

                return features, self.label
            if self.data_label_cur_number != curindex:
                self.copydf2 = self.copydf.copy()
                field = self.ui_interfacewin.comboBox_model_label.currentText()
                self.label = self.copydf2[field]
                self.copydf2 = self.copydf2.drop(field, axis=1)
                features = self.copydf2
                self.data_label_cur_number = self.ui_interfacewin.comboBox_model_label.currentIndex()
                return features, self.label

        # 切割数据集
        def split_data2(self):
            if self.is_split_data == False and self.data_label_cur_number == None:
                self.copydf2 = self.copydf.copy()
                field = self.ui_interfacewin.comboBox_model_label.currentText()
                self.label = self.copydf2[field]
                self.label_count = len(self.label.unique())

                # fieldunique = np.array(self.copydf[field].unique())
                lb_encoder = LabelEncoder()  # 独热编码，开始
                lb_encoder = lb_encoder.fit(np.array(self.label))
                lb_trans_f = lb_encoder.transform(np.array(self.label))
                oht_enoder = OneHotEncoder().fit(lb_trans_f.reshape(-1, 1))
                self.copydf[field] = lb_trans_f
                self.result2 = pd.DataFrame(oht_enoder.transform(lb_trans_f.reshape(-1, 1)).toarray())
                # self.result2 = result.toarray()
                self.result2 = self.result2.apply(lambda x: list(x), axis=1)

                self.copydf2 = self.copydf2.drop(field, axis=1)
                features = self.copydf2
                self.is_split_data = True
                self.data_label_cur_number = self.ui_interfacewin.comboBox_model_label.currentIndex()
                return features, self.result2, self.label_count
            if self.is_split_data and self.data_label_cur_number != None:
                curindex = self.ui_interfacewin.comboBox_model_label.currentIndex()
                if self.data_label_cur_number == curindex:
                    features = self.copydf2
                    self.label = self.label
                    label_count = self.label_count
                    return features, self.result2, label_count
                if self.data_label_cur_number != curindex:
                    self.copydf2 = self.copydf.copy()
                    field = self.ui_interfacewin.comboBox_model_label.currentText()
                    self.label = self.copydf2[field]
                    self.label_count = len(self.label.unique())
                    # fieldunique = np.array(self.copydf[field].unique())
                    lb_encoder = LabelEncoder()  # 独热编码，开始
                    lb_encoder = lb_encoder.fit(np.array(self.label))
                    lb_trans_f = lb_encoder.transform(np.array(self.label))
                    oht_enoder = OneHotEncoder().fit(lb_trans_f.reshape(-1, 1))
                    self.copydf[field] = lb_trans_f
                    self.result2 = pd.DataFrame(oht_enoder.transform(lb_trans_f.reshape(-1, 1)).toarray())
                    # self.result2 = result.toarray()
                    self.copydf2 = self.copydf2.drop(field, axis=1)
                    features = self.copydf2
                    self.data_label_cur_number = self.ui_interfacewin.comboBox_model_label.currentIndex()
                    return features, self.result2, self.label_count

    #切分数据集
    def cut_data(self):
        features, label = self.split_data()

        f_v = features.values
        f_names = features.columns.values
        l_v = label.values
        X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
        return X_train, X_test, Y_train, Y_test, X_validation, Y_validation

    #生成模型,评测模型
    def produce_evaluation_model(self, clf, name):
        X_train, X_test, Y_train, Y_test, X_validation, Y_validation = self.cut_data()
        clf.fit(X_train, Y_train)
        xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
        """
        if self.ui_interfacewin.checkBox_print_model_pdf.isChecked():
            dot_data = StringIO()
            f_names = self.copydf.columns.values
            export_graphviz(clf, out_file=dot_data, feature_names=f_names, class_names=["NL", "L"], filled=True,
                            rounded=True, special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_pdf("dt_tree_2.pdf")
        """
        ok = QMessageBox.question(self, "提示", "是否保存模型?", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            joblib.dump(clf, 'model/%s_model' % (name))
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)  # 验证

            self.ui_interfacewin.textEdit_result.append("%s-ACC:%s" % (name, str(accuracy_score(Y_part, Y_pred))))
            self.ui_interfacewin.textEdit_result.append("%s-REC:%s" % (name, str(recall_score(Y_part, Y_pred))))
            self.ui_interfacewin.textEdit_result.append("%s-F1:%s" % (name, str(f1_score(Y_part, Y_pred))))

    #保存模型
    def save_model_pdf(self, clf):
        dot_data = StringIO()
        f_names = self.copydf.columns.values
        export_graphviz(clf, out_file=dot_data, feature_names=f_names, class_names=["NL", "L"], filled=True,
                        rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("dt_tree_2.pdf")

    #KNN
    @pyqtSlot()
    def on_model_classifier_KNN_clicked(self):
        n_neighbors = self.ui_interfacewin.spinBox_classifier_KNN_n_neighbors.value()
        weights = self.ui_interfacewin.comboBox__classifier_KNN_weights.currentText()
        algorithm = self.ui_interfacewin.comboBox_classifier_KNN_algorithm.currentText()
        leaf_size = self.ui_interfacewin.spinBox_classifier_KNN_leaf_size.value()
        p = self.ui_interfacewin.spinBox_classifier_KNN_P.value()
        metric = self.ui_interfacewin.comboBox_classifier_KNN_metric.currentText()
        n_jobs = self.ui_interfacewin.spinBox_classifier_KNN_n_jobs.value()
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p,
                                   metric=metric, metric_params=None ,n_jobs=n_jobs)
        name = 'KNN'
        self.produce_evaluation_model(clf=clf, name=name)

    #GaussianNB
    @pyqtSlot()
    def on_model_classifier_GaussianNB_clicked(self):
        clf = GaussianNB(priors=None, var_smoothing=1e-09)
        name = 'GaussianNB'
        self.produce_evaluation_model(clf=clf, name=name)
    #BernoulliNB
    @pyqtSlot()
    def on_model_classifier_BernoulliNB_clicked(self):
        alpha = self.ui_interfacewin.doubleSpinBox_classifier_BernoulliNB_alpha.value()
        binarize = self.ui_interfacewin.doubleSpinBox_classifier_BernoulliNB_binarize.value()
        fit_prior = False
        if self.ui_interfacewin.checkBox_classifier_BernoulliNB_fit_prior.isChecked():
            fit_prior = True
        clf = BernoulliNB(alpha=alpha, binarize=binarize, fit_prior=fit_prior, class_prior=None)
        name = 'BernoulliNB'
        self.produce_evaluation_model(clf=clf, name=name)
    #DecisionTreeGini
    @pyqtSlot()
    def on_model_classifier_DecisionTreeGini_clicked(self):
        ################################################max_features
        ################################################class_weight
        ################################################
        criterion = self.ui_interfacewin.comboBox_classifier_DecisionTreeGini_criterion.currentText()
        splitter = self.ui_interfacewin.comboBox_classifier_DecisionTreeGini_splitter.currentText()
        max_depth = None
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_max_depth.isChecked():
            max_depth = self.ui_interfacewin.spinBox_classifier_DecisionTreeGini_max_depth.value()
        min_samples_split = self.ui_interfacewin.spinBox_classifier_DecisionTreeGini_min_samples_split.value()
        min_samples_leaf = self.ui_interfacewin.spinBox_classifier_DecisionTreeGini_min_samples_leaf.value()
        min_weight_fraction_leaf = self.ui_interfacewin.doubleSpinBox_classifier_DecisionTreeGini_min_weight_fraction_leaf.value()
        min_impurity_decrease = self.ui_interfacewin.doubleSpinBox_classifier_DecisionTreeGini_min_impurity_decrease.value()
        max_leaf_nodes = None
        if self.ui_interfacewin.checkBox_classifier_DecisionTreeGini_max_leaf_nodes.isChecked():
            max_leaf_nodes = self.ui_interfacewin.spinBox_classifier_DecisionTreeGini_max_leaf_nodes.value()
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_DecisionTreeGini_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_DecisionTreeGini_random_state.value()
        presort = False
        if self.ui_interfacewin.checkBox_classifier_DecisionTreeGini_presort.isChecked():
            presort = True
        clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=None,
                                    random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, class_weight=None, presort=presort)
        name = 'DecisionTreeGini'
        self.produce_evaluation_model(clf=clf, name=name)
    #DecisionTreeEntropy
    @pyqtSlot()
    def on_model_classifier_DecisionTreeEntropy_clicked(self):
        #################################多余的  ‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’‘’
        clf = DecisionTreeClassifier(criterion="entropy")
        name = 'DecisionTreeEntropy'
        self.produce_evaluation_model(clf=clf, name=name)
    #SVM Classifier
    @pyqtSlot()
    def on_model_classifier_SVM_Classifier_clicked(self):
        ###############################################tol
        ###############################################class_weight
        C = self.ui_interfacewin.doubleSpinBox_classifier_SVM_SVC_C.value()
        kernel = self.ui_interfacewin.comboBox_classifier_SVM_SVC_kernel.currentText()
        degree = self.ui_interfacewin.spinBox_classifier_SVM_SVC_degree.value()
        gamma = 'auto'
        if self.ui_interfacewin.checkBox_classifier_SVM_SVC_gamma.isChecked():
            gamma = self.ui_interfacewin.doubleSpinBox_classifier_SVM_SVC_gamma.value()
        coef0 = self.ui_interfacewin.doubleSpinBox_classifier_SVM_SVC_coef0.value()
        shrinking = False
        if self.ui_interfacewin.checkBox_classifier_SVM_SVC_shrinking.isChecked():
            shrinking = True
        probability =False
        if self.ui_interfacewin.checkBox_classifier_SVM_SVC_probability.isChecked():
            probability = True
        cache_size = self.ui_interfacewin.doubleSpinBox_classifier_SVM_SVC_cache_size.value()
        verbose = False
        if self.ui_interfacewin.checkBox_classifier_SVM_SVC_verbose.isChecked():
            verbose = True
        max_iter = self.ui_interfacewin.spinBox_classifier_SVM_SVC_max_iter.value()
        decision_function_shape = self.ui_interfacewin.comboBox_classifier_SVM_SVC_decision_function_shape.currentText()
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_SVM_SVC_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_SVM_SVC_random_state.value()
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=0.001, cache_size=cache_size,
                  class_weight=None, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, random_state=random_state)
        name = 'SVM Classifier'
        self.produce_evaluation_model(clf=clf, name=name)
    #OriginalRandomForest
    @pyqtSlot()
    def on_model_classifier_OriginalRandomForest_clicked(self):
        ###############################################max_features
        ###############################################class_weight
        n_estimators = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_n_estimators.value()
        criterion = self.ui_interfacewin.comboBox_classifier_OriginalRandomForest_criterion.currentText()
        max_depth = None
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_max_depth.isChecked():
            max_depth = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_max_depth.value()
        min_samples_split = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_min_samples_split.value()
        min_samples_leaf = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_min_samples_leaf.value()
        min_weight_fraction_leaf = self.ui_interfacewin.doubleSpinBox_classifier_OriginalRandomForest_min_weight_fraction_leaf.value()
        max_leaf_nodes = None
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_max_leaf_nodes.isChecked():
            max_leaf_nodes = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_max_leaf_nodes.value()
        min_impurity_decrease = self.ui_interfacewin.doubleSpinBox_classifier_OriginalRandomForest_min_impurity_decrease.value()
        bootstrap = False
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_bootstrap.isChecked():
            bootstrap = True
        oob_score = False
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_oob_score.isChecked():
            oob_score = True
        n_jobs = None
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_n_jobs.isChecked():
            n_jobs = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_n_jobs.value()
        verbose = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_verbose.value()
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_OriginalRandomForest_random_state.value()
        warm_start = False
        if self.ui_interfacewin.checkBox_classifier_OriginalRandomForest_warm_start.isChecked():
            warm_start = True
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features='auto',
                                     max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                                     oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                                     class_weight=None)
        name='OriginalRandomForest'
        self.produce_evaluation_model(clf=clf, name=name)
    #RandomForest
    @pyqtSlot()
    def on_model_classifier_RandomForest_clicked(self):
        ########################################################重复///////////////////////////////////////////////////////////
        clf = RandomForestClassifier(n_estimators=11, max_features=None)
        name = 'RandomForest'
        self.produce_evaluation_model(clf=clf, name=name)
    #Adaboost
    @pyqtSlot()
    def on_model_classifier_Adaboost_clicked(self):
        ############################################################base_estimator
        n_estimators = self.ui_interfacewin.spinBox_classifier_Adaboost_n_estimators.value()
        learning_rate = self.ui_interfacewin.doubleSpinBox_classifier_Adaboost_learning_rate.value()
        algorithm = self.ui_interfacewin.comboBox_classifier_Adaboost_algorithm.currentText()
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_Adaboost_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_Adaboost_random_state.value()
        clf = AdaBoostClassifier(base_estimator=None, n_estimators=n_estimators, learning_rate=learning_rate,
                                 algorithm=algorithm, random_state=random_state)
        name = 'Adaboost'
        self.produce_evaluation_model(clf=clf, name=name)
    #LogisticRegression
    @pyqtSlot()
    def on_model_classifier_LogisticRegression_clicked(self):
        ##################################################class_weight
        ##################################################l1_ratio=None
        penalty = self.ui_interfacewin.comboBox_classifier_LogisticRegression_penalty.currentText()
        dual = False
        if self.ui_interfacewin.checkBox_classifier_LogisticRegression_dual.isChecked():
            dual = True
        C = self.ui_interfacewin.doubleSpinBox_classifier_LogisticRegression_C.value()
        tol = self.ui_interfacewin.doubleSpinBox_classifier_LogisticRegression_tol.value()
        fit_intercept = False
        if self.ui_interfacewin.checkBox_classifier_LogisticRegression_fit_intercept.isChecked():
            fit_intercept = True
        solver = self.ui_interfacewin.comboBox_classifier_LogisticRegression_solver.currentText()
        inercept_scaling = self.ui_interfacewin.spinBox_classifier_LogisticRegression_intercept_scaling.value()
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_LogisticRegression_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_LogisticRegression_random_state.value()
        max_iter = self.ui_interfacewin.spinBox_classifier_LogisticRegression_max_iter.value()
        multi_class = self.ui_interfacewin.comboBox_classifier_LogisticRegression_multi_class.currentText()
        verbose = self.ui_interfacewin.spinBox_classifier_LogisticRegression_verbose.value()
        warm_start = False
        if self.ui_interfacewin.checkBox_classifier_LogisticRegression_warm_start.isChecked():
            warm_start = True
        n_jobs = None
        if self.ui_interfacewin.checkBox_classifier_LogisticRegression_n_jobs.isChecked():
            n_jobs = self.ui_interfacewin.spinBox_classifier_LogisticRegression_n_jobs.value()
        l1_ratio = None
        if self.ui_interfacewin.checkBox_classifier_LogisticRegression_l1_ratio.isChecked():
            l1_ratio = self.ui_interfacewin.doubleSpinBox_classifier_LogisticRegression_l1_ratio.value()
        clf = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=inercept_scaling,
                                 class_weight=None, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class,
                                 verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        name = 'LogisticRegression'
        self.produce_evaluation_model(clf=clf, name=name)

    #GBDT
    @pyqtSlot()
    def on_model_classifier_GBDT_clicked(self):
        ######################################presort
        ######################################verbose
        ######################################max_features
        #######################################init
        loss = self.ui_interfacewin.comboBox_classifier_GBDT_loss.currentText()
        learning_rate = self.ui_interfacewin.doubleSpinBox_classifier_GBDT_learning_rate.value()
        n_estimators = self.ui_interfacewin.spinBox_classifier_GBDT_n_estimators.value()
        subsample = self.ui_interfacewin.doubleSpinBox_classifier_GBDT_subsample.value()
        max_depth = self.ui_interfacewin.spinBox_classifier_GBDT_max_depth.value()
        min_impurity_decrease = self.ui_interfacewin.doubleSpinBox_classifier_GBDT_min_impurity_decrease.value()
        criterion = self.ui_interfacewin.comboBox_classifier_GBDT_criterion.currentText()
        min_samples_split = self.ui_interfacewin.spinBox_classifier_GBDT_min_samples_split.value()
        min_samples_leaf = self.ui_interfacewin.spinBox_classifier_GBDT_min_samples_leaf.value()
        min_weight_fraction_leaf = self.ui_interfacewin.doubleSpinBox_classifier_DecisionTreeGini_min_weight_fraction_leaf.value()
        warm_start = False
        if self.ui_interfacewin.checkBox_classifier_GBDT_warm_start.isChecked():
            warm_start = True
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_GBDT_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_DecisionTreeGini_random_state.value()
        max_leaf_nodes = None
        if self.ui_interfacewin.checkBox_classifier_GBDT_max_leaf_nodes.isChecked():
            max_leaf_nodes = self.ui_interfacewin.spinBox_classifier_GBDT_max_leaf_nodes.value()
        validation_fraction = self.ui_interfacewin.doubleSpinBox_classifier_GBDT_validation_fraction.value()
        n_iter_no_change = None
        if self.ui_interfacewin.checkBox_classifier_GBDT_n_iter_no_change.isChecked():
            n_iter_no_change = self.ui_interfacewin.spinBox_classifier_GBDT_n_iter_no_change.value()
        tol = self.ui_interfacewin.doubleSpinBox_classifier_GBDT_tol.value()
        clf = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, criterion=criterion,
                                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                                         max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, init=None, random_state=random_state, max_features=None,
                                         verbose=0, max_leaf_nodes=None, warm_start=warm_start, presort='auto', validation_fraction=validation_fraction,
                                         n_iter_no_change=n_iter_no_change, tol=tol)
        name = 'GBDT'
        self.produce_evaluation_model(clf=clf, name=name)

    #linear_model拟合评价
    def linear_model_fit_evaluation(self, regr, name):
        features, label = self.split_data()
        regr.fit(features.values, label.values)
        ok = QMessageBox.question(self, "提示", "是否保存模型?", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            joblib.dump(regr, 'model/%s_model' % (name))
        Y_pred = regr.predict(features.values)
        self.ui_interfacewin.textEdit_result.append("%s-Coef:%s" % (name, str(regr.coef_)))
        self.ui_interfacewin.textEdit_result.append(
            "%s-MSE:%s" % (name, str(mean_squared_error(label.values, Y_pred))))
        self.ui_interfacewin.textEdit_result.append(
            "%s-MAE:%s" % (name, str(mean_absolute_error(label.values, Y_pred))))
        self.ui_interfacewin.textEdit_result.append("%s-r2_score:%s" % (name, str(r2_score(label.values, Y_pred))))

    #LinearRegression
    @pyqtSlot()
    def on_model_classifier_LinearRegression_clicked(self):
        fit_intercept = True
        if self.ui_interfacewin.checkBox_classifier_LinearRegression_fit_intercept.isChecked():
            fit_intercept = True
        normalize = True
        if self.ui_interfacewin.checkBox_classifier_LinearRegression_normalize.isChecked():
            normalize = True
        copy_X = False
        if self.ui_interfacewin.checkBox_classifier_LinearRegression_copy_X.isChecked():
            copy_X = True
        n_jobs = None
        if self.ui_interfacewin.checkBox_classifier_LinearRegression_n_jobs.isChecked():
            n_jobs = self.ui_interfacewin.spinBox_classifier_LinearRegression_n_jobs.value()
        regr = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
        name = 'LinearRegression'
        self.linear_model_fit_evaluation(regr=regr, name=name)
    #Ridge
    @pyqtSlot()
    def on_model_classifier_Ridge_clicked(self):
        alpha = self.ui_interfacewin.doubleSpinBox_classifier_Ridge_alpha.value()
        fit_intercept = False
        if self.ui_interfacewin.checkBox_classifier_Ridge_fit_intercept.isChecked():
            fit_intercept = True
        normalize = False
        if self.ui_interfacewin.checkBox_classifier_Ridge_normalize.isChecked():
            normalize = True
        copy_X = False
        if self.ui_interfacewin.checkBox_classifier_Ridge_copy_X.isChecked():
            copy_X = True
        max_iter = self.ui_interfacewin.spinBox_classifier_Ridge_max_iter.value()
        tol = self.ui_interfacewin.doubleSpinBox_classifier_Ridge_tol.value()
        solver = self.ui_interfacewin.comboBox_classifier_Ridge_solver.currentText()
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_Ridge_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_Ridge_random_state.value()
        regr = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
                                         max_iter=max_iter, tol=tol, solver=solver, random_state=random_state)
        name = 'Ridge'
        self.linear_model_fit_evaluation(regr=regr, name=name)
    #Lasso
    @pyqtSlot()
    def on_model_classifier_Lasso_clicked(self):
        alpha = self.ui_interfacewin.doubleSpinBox_classifier_Lasso_alpha.value()
        fit_intercept = False
        if self.ui_interfacewin.checkBox_classifier_Lasso_fit_intercept.isChecked():
            fit_intercept = True
        normalize = False
        if self.ui_interfacewin.checkBox_classifier_Lasso_normalize.isChecked():
            normalize = True
        copy_X = False
        if self.ui_interfacewin.checkBox_classifier_Lasso_copy_X.isChecked():
            copy_X = True
        precompute = False
        if self.ui_interfacewin.checkBox_classifier_Lasso_precompute.isChecked():
            precompute = True
        max_iter = self.ui_interfacewin.spinBox_classifier_Lasso_max_iter.value()
        tol = self.ui_interfacewin.doubleSpinBox_classifier_Lasso_tol.value()
        solver = self.ui_interfacewin.comboBox_classifier_Lasso_solver.currentText()
        random_state = None
        if self.ui_interfacewin.checkBox_classifier_Lasso_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_classifier_Lasso_random_state.value()
        warm_start = False
        if self.ui_interfacewin.checkBox_classifier_Lasso_warm_start.isChecked():
            warm_start = True
        positive = False
        if self.ui_interfacewin.checkBox_classifier_Lasso_positive.isChecked():
            positive = True
        selection = self.ui_interfacewin.comboBox_classifier_Lasso_selection.currentText()

        regr = Lasso(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, precompute=precompute, copy_X=copy_X,
                     max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection)
        name = 'Lasso'
        self.linear_model_fit_evaluation(regr=regr, name=name)

    #半监督
    #添加标注
    @pyqtSlot(int)
    def on_comboBox_model_label_currentIndexChanged(self, index):
        field = self.ui_interfacewin.comboBox_model_label.currentText()
        if len(field) != 0:
            field_unique_values = self.copydf[field].unique()
            str2 = field_unique_values.tolist()
            new_str2 = [str(x) for x in str2]
            self.ui_interfacewin.comboBox_semi_supervised_learning_label.clear()
            self.ui_interfacewin.comboBox_semi_supervised_learning_label.addItems(new_str2)

    def get_semi_supervised_learning_label(self):
        field = self.ui_interfacewin.comboBox_model_label.currentText()
        self.field2 = self.ui_interfacewin.comboBox_semi_supervised_learning_label.currentText()
        if self.is_semi_supervised_learning_data_split == False:
            self.copydf_semi_supervised = self.copydf.copy()
            if is_number(self.field2):
                self.field2 = float(self.field2)
                if self.field2 == 0:
                    self.copydf_semi_supervised[field] = self.copydf_semi_supervised[field] + 1
                    self.field2 = 1
                    self.is_zero_field2 = True
                    #print(self.copydf_semi_supervised)
            self.label = self.copydf_semi_supervised[field]

            self.label2 = self.label[self.label == self.field2]
            self.X_label = self.copydf_semi_supervised[self.label == self.field2]
            self.X_label = self.X_label.drop(field, axis=1)
            self.copydf_semi_supervised = self.copydf_semi_supervised.drop(field, axis=1)
            self.is_semi_supervised_learning_data_split = True
            self.cur_semi_supervised_learning_field_index = self.ui_interfacewin.comboBox_model_label.currentIndex
            self.cur_semi_supervised_learning_fieldvalue_index = self.ui_interfacewin.comboBox_semi_supervised_learning_label.currentIndex()
            return self.copydf_semi_supervised, self.label, self.label2, self.X_label, self.field2
        if self.is_semi_supervised_learning_data_split:
            cur_field_index = self.ui_interfacewin.comboBox_model_label.currentIndex()
            if cur_field_index == self.cur_semi_supervised_learning_field_index:
                cur_fieldvalues_index = self.ui_interfacewin.comboBox_semi_supervised_learning_label.currentIndex()
                if cur_fieldvalues_index == self.cur_semi_supervised_learning_fieldvalue_index:
                    return self.copydf_semi_supervised, self.label, self.label2, self.X_label, self.field2
                if cur_fieldvalues_index != self.cur_semi_supervised_learning_fieldvalue_index:
                    self.copydf_semi_supervised = self.copydf.copy()
                    if is_number(self.field2):
                        self.field2 = float(self.field2)
                        if self.field2 == 0:
                            self.copydf_semi_supervised = self.copydf_semi_supervised[self.field2] + 1
                            self.field2 = 1
                            self.is_zero_field2 = True
                    self.label = self.copydf_semi_supervised[field]

                    self.label2 = self.label[self.label == self.field2]
                    self.X_label = self.copydf_semi_supervised[self.label == self.field2]
                    self.X_label = self.X_label.drop(field, axis=1)
                    self.copydf_semi_supervised = self.copydf_semi_supervised.drop(field, axis=1)
                    self.is_semi_supervised_learning_data_split = True
                    self.cur_semi_supervised_learning_field_index = self.ui_interfacewin.comboBox_model_label.currentIndex
                    self.cur_semi_supervised_learning_fieldvalue_index = self.ui_interfacewin.comboBox_semi_supervised_learning_label.currentIndex()
                    return self.copydf_semi_supervised, self.label, self.label2, self.X_label, self.field2
            if cur_field_index != self.cur_semi_supervised_learning_field_index:
                self.copydf_semi_supervised = self.copydf.copy()
                if is_number(self.field2):
                    self.field2 = float(self.field2)
                    if self.field2 == 0:
                        self.copydf_semi_supervised = self.copydf_semi_supervised[self.field2] + 1
                        self.field2 = 1
                        self.is_zero_field2 = True
                self.label = self.copydf_semi_supervised[field]

                self.label2 = self.label[self.label == self.field2]
                self.X_label = self.copydf_semi_supervised[self.label == self.field2]
                self.X_label = self.X_label.drop(field, axis=1)
                self.copydf_semi_supervised = self.copydf_semi_supervised.drop(field, axis=1)
                self.is_semi_supervised_learning_data_split = True
                self.cur_semi_supervised_learning_field_index = self.ui_interfacewin.comboBox_model_label.currentIndex
                self.cur_semi_supervised_learning_fieldvalue_index = self.ui_interfacewin.comboBox_semi_supervised_learning_label.currentIndex()
                return self.copydf_semi_supervised, self.label, self.label2, self.X_label, self.field2
    @pyqtSlot()
    def on_semi_supervised_learning_Btn_clicked(self):
        kernel = self.ui_interfacewin.comboBox_supervised_learning_kernel.currentText()
        gamma = self.ui_interfacewin.doubleSpinBox_supervised_learning_gamma.value()
        n_neighbors = self.ui_interfacewin.spinBox_supervised_learning_n_neighbors.value()
        max_iter = self.ui_interfacewin.spinBox_supervised_learning__max_iter.value()
        tol = self.ui_interfacewin.doubleSpinBox_supervised_learning_tol.value()
        n_jobs = None
        if self.ui_interfacewin.checkBox_supervised_learning_n_jobs.isChecked():
            n_jobs = self.ui_interfacewin.spinBox_supervised_learning_n_jobs.value()
        label_prop_model = LabelPropagation(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        features, label, label2, X_label, field2 = self.get_semi_supervised_learning_label()
        f_v = features.values
        #f_names = features.columns.values
        l_v = label.values
        X_tt, X_validation, Y_tt, Y_validation = train_test_split(f_v, l_v, test_size=0.2)
        X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)
        label_prop_model.fit(X_train, Y_train)
        name = 'LabelPropagation'
        ok = QMessageBox.question(self, "提示", "是否保存模型?", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            joblib.dump(label_prop_model, 'model/%s_model' % (name))

        Y_pred = label_prop_model.predict(X_test)
        self.ui_interfacewin.textEdit_result.append("%s-test-ACC:%s" % (name, str(accuracy_score(Y_test, Y_pred))))
        self.ui_interfacewin.textEdit_result.append("%s-test-REC:%s" % (name, str(recall_score(Y_test, Y_pred))))
        self.ui_interfacewin.textEdit_result.append("%s-test-F1:%s" % (name, str(f1_score(Y_test, Y_pred))))
        Y_pred2 = label_prop_model.predict(X_validation)
        self.ui_interfacewin.textEdit_result.append("%s-validation-ACC:%s" % (name, str(accuracy_score(Y_validation, Y_pred2))))
        self.ui_interfacewin.textEdit_result.append("%s-validation-REC:%s" % (name, str(recall_score(Y_validation, Y_pred2))))
        self.ui_interfacewin.textEdit_result.append("%s-validation-F1:%s" % (name, str(f1_score(Y_validation, Y_pred2))))
        Y_pred3 = label_prop_model.predict(X_label)

        if self.is_zero_field2:
            field2 = field2 - 1
        self.ui_interfacewin.textEdit_result.append(
            "%s-%s-ACC:%s" % (name, str(field2), str(accuracy_score(label2, Y_pred3))))
        self.ui_interfacewin.textEdit_result.append(
            "%s-%s-REC:%s" % (name, str(field2), str(recall_score(label2, Y_pred3, labels=np.unique(Y_pred3)))))
        self.ui_interfacewin.textEdit_result.append(
            "%s-%s-F1:%s" % (name, str(field2), str(f1_score(label2, Y_pred3, labels=np.unique(Y_pred3)))))
        #label_prop_model.fit(features, label)
        #Y_pred = label_prop_model.predict(features)
        #Y_pred = Y_pred[label2]
        #self.ui_interfacewin.textEdit_result.append("LabelPropagation-ACC:%s" % (accuracy_score(label2, Y_pred)))
        #self.ui_interfacewin.textEdit_result.append("LabelPropagation-REC:%s" % (recall_score(label2, Y_pred, average="micro")))
        #self.ui_interfacewin.textEdit_result.append("LabelPropagation-F-Score:%s" % (f1_score(label2, Y_pred, average="micro")))

    #关联分析
    #添加项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_correlation_analysis_Apriori_itemClicked(self, item):
        itemtext = item.text()
        self.ui_interfacewin.listWidget_correlation_analysis_Apriori_2.addItem(itemtext)

    #删除项目
    @pyqtSlot(QListWidgetItem)
    def on_listWidget_correlation_analysis_Apriori_2_itemClicked(self, item):
        curindex = self.ui_interfacewin.listWidget_correlation_analysis_Apriori_2.currentIndex()
        currow = curindex.row()
        self.ui_interfacewin.listWidget_correlation_analysis_Apriori_2.takeItem(currow)

    @pyqtSlot()
    def on_correlation_analysis_Apriori_Btn_clicked(self):
        cur_cor_analyse_type = self.ui_interfacewin.comboBox_correlation_analysis_Apriori.currentIndex()
        if cur_cor_analyse_type == 0:
            dataSet = self.loadDataSet()
            minSupport = self.ui_interfacewin.doubleSpinBox_correlation_analysis_Apriori_minSupport.value()
            minConf=self.ui_interfacewin.doubleSpinBox_correlation_analysis_Apriori_minConf.value()
            #from apyori import apriori
            #result = list(apriori(transactions=dataSet, min_support=minSupport, min_confidence=minConf))
            #self.ui_interfacewin.textEdit_result.append("%s" % (result))
            L, suppData = self.apriori(dataSet, minSupport=minSupport)
            self.ui_interfacewin.textEdit_result.append("符合%s最小支持度的项集：\n%s\n全部项集:\n%s" % (minSupport, L, suppData))
            rules = self.generateRules3(L, suppData, minConf=minConf)
            self.ui_interfacewin.textEdit_result.append("最小可信度为%s的关联规则:\n%s" % (minConf, rules))
        if cur_cor_analyse_type == 1:
            simpDat = self.loadDataSet()
            minSup = self.ui_interfacewin.spinBox_correlation_analysis_FPGrowth.value()
            dataSet = fpGrowth.loadSimpDat(simpDat)
            freqItems= fpGrowth.fpGrowth(dataSet, minSup=minSup)
            #self.ui_interfacewin.textEdit_result.append("FP-growth树:\n%s" % (tree))
            self.ui_interfacewin.textEdit_result.append("FP-growth:\n%s" % (freqItems))
    #获取频繁项
    def loadDataSet(self):
        fieldcount = self.ui_interfacewin.listWidget_correlation_analysis_Apriori_2.count()
        fields = []
        for i in range(0, fieldcount):
            field = self.ui_interfacewin.listWidget_correlation_analysis_Apriori_2.item(i).text()
            fields.append(field)
        fieldvalues= self.copydf[fields]
        #fieldvalues2 = self.copydf[fields]
        col = fieldvalues.columns
        t = []
        d = []
        for i in range(len(col)):
            if i == 0:
                t = fieldvalues[col[i]].unique()
            if i == 1:
                f = fieldvalues[col[i]].unique()
                d = np.hstack((t, f))
            if i > 1:
                f = fieldvalues[col[i]].unique()
                d = np.hstack((d, f))
        unique_field = pd.Series(d)
        unique_field = unique_field.dropna()
        unique_field = unique_field.unique()
        label_field = LabelEncoder()
        label_field_fit = label_field.fit(unique_field)
        field_label_result = []
        for index, row in fieldvalues.iterrows():
            row = row.dropna()
            row = label_field_fit.transform(row)
            row = row.tolist()
            field_label_result.append(row)

        #return field_label_result
        #return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
        return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

    def createC1(self, dataSet):
        C1 = []
        for transaction in dataSet:
            for item in transaction:
                if not [item] in C1:
                    C1.append([item])
        C1.sort()
        return list(map(frozenset, C1))

    def scanD(self, D, CK, minSupport):
        ssCnt = {}
        for tid in D:
            for can in CK:
                if can.issubset(tid):
                    if can not in ssCnt:
                        ssCnt[can] = 1
                    else:
                        ssCnt[can] += 1
        numItems = float(len(D))
        retList = []
        supportData = {}
        for key in ssCnt:
            support = ssCnt[key] / numItems
            if support >= minSupport:
                retList.insert(0, key)
            supportData[key] = support
        return retList, supportData

    def aprioriGen(self, LK, K):
        retList = []
        lenLK = len(LK)
        for i in range(lenLK):
            for j in range(i+1, lenLK):
                L1 = list(LK[i])[:K-2]
                L2 = list(LK[j])[:K-2]
                L1.sort(); L2.sort()
                if L1 == L2:
                    retList.append(LK[i] | LK[j])
        return retList

    def apriori(self, dataSet, minSupport):
        D = list(map(set, dataSet))
        C1 = self.createC1(dataSet)
        L1, supportData = self.scanD(D, C1, minSupport)
        L = [L1]
        K = 2
        while(len(L[K-2]) > 0):
            CK = self.aprioriGen(L[K-2], K)
            LK, supK = self.scanD(D, CK, minSupport)
            supportData.update(supK)
            L.append(LK)
            K += 1
        return L, supportData

    #获取关联规则
    """
    def generateRules(L, supportData, minConf):
        bigRuleList = []
        for i in range(1, len(L)):
            for freqSet in L[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if (i > 1):
                    # 三个及以上元素的集合
                    self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
                else:
                    # 两个元素的集合
                    self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)
        return bigRuleList
    
    def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
        ''' 生成候选规则集 '''
        m = len(H[0])
        if (len(freqSet) > (m + 1)):
            Hmpl = self.aprioriGen(H, m + 1)
            Hmpl = self.calcConf(freqSet, Hmpl, supportData, brl, minConf)
            if (len(Hmpl) > 1):
                self.rulesFromConseq(freqSet, Hmpl, supportData, brl, minConf)

    def generateRules2(L, supportData, minConf=0.7):
        bigRuleList = []
        for i in range(1, len(L)):
            for freqSet in L[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if (i > 1):
                    # 三个及以上元素的集合
                    H1 = self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)
                    self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
                else:
                    # 两个元素的集合
                    self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)
        return bigRuleList
    """
    def generateRules3(self, L, supportData, minConf):
        bigRuleList = []
        for i in range(1, len(L)):
            for freqSet in L[i]:
                    H1 = [frozenset([item]) for item in freqSet]
                    self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
        return bigRuleList

    def calcConf(self, freqSet, H, supportData, brl, minConf):
        prunedH = []
        for conseq in H:
            conf = supportData[freqSet]/supportData[freqSet-conseq]
            if conf >= minConf:
                self.ui_interfacewin.textEdit_result.append("%s,'-->',conseq,'conf:',%s" % (freqSet-conseq, conf))
                brl.append((freqSet-conseq, conseq, conf))
                prunedH.append(conseq)
        return prunedH


    def rulesFromConseq(self, freqSet, H, supportData, brl, minConf):
        m = len(H[0])
        while(len(freqSet) > m):
            H = self.calcConf(freqSet, H, supportData, brl, minConf)
            if(len(freqSet) > (m)):
                Hmp1 = self.calcConf(freqSet, H, supportData, brl, minConf)
                if (len(H) > 1):
                    H = self.aprioriGen(H, m+1)
                    m += 1
                else:
                    break

    #聚类
    @pyqtSlot()
    def on_cluster_KMeans_Btn_2_clcked(self):
        #################################nit ： {'k-means ++'，'random'或ndarray}
        n_clusters = self.ui_interfacewin.spinBox_cluster_KMeans_n_clusters.value()
        init = self.ui_interfacewin.comboBox_cluster_KMeans_init.currentText()
        n_init = self.ui_interfacewin.spinBox_cluster_KMeans_n_init.value()
        max_iter = self.ui_interfacewin.spinBox_cluster_KMeans_max_iter.value()
        tol = self.ui_interfacewin.doubleSpinBox_cluster_KMeans_tol.value()
        random_state = None
        if self.ui_interfacewin.checkBox_cluster_KMeans_random_state.isChecked():
            random_state = self.ui_interfacewin.spinBox_cluster_KMeans_random_state.value()
        copy_x = False
        if self.ui_interfacewin.checkBox_cluster_KMeans_copy_x.isChecked():
            copy_x = True
        n_jobs = None
        if self.ui_interfacewin.checkBox_cluster_KMeans_n_jobs.isChecked():
            n_jobs = self.ui_interfacewin.spinBox_cluster_KMeans_n_jobs.value()
        algorithm = self.ui_interfacewin.comboBox_cluster_KMeans_algorithm.currentText()

        if self.ui_interfacewin.comboBox_cluster_KMeans_precompute_distances.currentIndex() == 0:
            precompute_distances = 'auto'
        if self.ui_interfacewin.comboBox_cluster_KMeans_precompute_distances.currentIndex() == 1:
            precompute_distances = False
        if self.ui_interfacewin.comboBox_cluster_KMeans_precompute_distances.currentIndex() == 2:
            precompute_distances = True
        X = self.copydf
        km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol, precompute_distances=precompute_distances,
                    verbose=0, random_state=random_state, copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm).fit(X)
        ok = QMessageBox.question(self, "提示", "是否保存模型?", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            joblib.dump(km, 'model/KMeans_model')

        self.ui_interfacewin.textEdit_result.append("KMeans_labels:\n%s" % (km.labels_))
    @pyqtSlot()
    def on_cluster_DBSCAN_Btn_clicked(self):
        ######################################metric_params
        eps = self.ui_interfacewin.doubleSpinBox_DBSCAN_eps.value()
        min_samples = self.ui_interfacewin.spinBox_DBSCAN_min_samples.value()
        metric = self.ui_interfacewin.comboBox_DBSCAN_metric.currentText()
        algorithm = self.ui_interfacewin.comboBox_DBSCAN_algorithm.currentText()
        leaf_size = self.ui_interfacewin.spinBox_DBSCAN_leaf_size.value()
        p = self.ui_interfacewin.doubleSpinBox_DBSCAN_p.value()
        n_jobs = None
        if self.ui_interfacewin.checkBox_DBSCAN_n_jobs.isChecked():
            n_jobs = self.ui_interfacewin.spinBox_DBSCAN_n_jobs.value()
        X = self.copydf
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=None, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs).fit(X)
        ok = QMessageBox.question(self, "提示", "是否保存模型?", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            joblib.dump(db, 'model/DBSCAN_model')

        self.ui_interfacewin.textEdit_result.append("DBSCAN_labels_:\n%s" % (db.labels_))
    @pyqtSlot()
    def on_cluster_Agglomerative_Btn_clicked(self):
        ##################################################memory
        #################################################connectivity : array-like or callable, optional
        n_clusters = None
        if self.ui_interfacewin.checkBox_cluster_Agglomerative_n_clusters.isChecked():
            n_clusters = self.ui_interfacewin.spinBox_cluster_Agglomerative_n_clusters.value()
        affinity = self.ui_interfacewin.comboBox_cluster_Agglomerative_affinity.currentText()
        distance_threshold = None
        if self.ui_interfacewin.checkBox_cluster_Agglomerative_distance_threshold.isChecked():
            distance_threshold = self.ui_interfacewin.doubleSpinBox_cluster_Agglomerative_distance_threshold.value()
        if self.ui_interfacewin.comboBox_cluster_Agglomerative_compute_full_tree.currentIndex() == 0:
            compute_full_tree = 'auto'
        if self.ui_interfacewin.comboBox_cluster_Agglomerative_compute_full_tree.currentIndex() == 1:
            compute_full_tree = True
        if self.ui_interfacewin.comboBox_cluster_Agglomerative_compute_full_tree.currentIndex() == 2:
            compute_full_tree = False
        linkage = self.ui_interfacewin.comboBox_cluster_Agglomerative_linkage.currentText()
        X = self.copydf
        clu = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, memory=None, connectivity=None, compute_full_tree=compute_full_tree,
                                      linkage=linkage, distance_threshold=distance_threshold).fit(X)
        ok = QMessageBox.question(self, "提示", "是否保存模型?", QMessageBox.Ok, QMessageBox.Cancel)
        if ok == QMessageBox.Ok:
            joblib.dump(clu, 'model/AgglomerativeClustering_model')

        self.ui_interfacewin.textEdit_result.append("AgglomerativeClustering_labels:%s" % (clu.labels_))

    #绘图
    @pyqtSlot()
    def on_pushButton_clicked(self):
       # self.plotly_show.show()
        x = self.copydf["satisfaction_level"].index
        y = self.copydf["time_spend_company"]
        y1 = self.copydf["satisfaction_level"]
        y2 = self.copydf["average_monthly_hours"]
        self.plotly_show.show_scatter_plots(x=x, y=y, x1=x, y1=y1, x2=x, y2=y2)
#可以检测小数，负数
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
     pass
    try:
     import unicodedata
     unicodedata.numeric(s)
     return True
    except (TypeError, ValueError):
     pass
    return False

#self.ui_interfacewin.textEdit_result.setText(ttt)
if __name__=="__main__":
    app = QApplication(sys.argv)
    interfacewin = InterfaceWin()
    interfacewin.show()

    sys.exit(app.exec_())
