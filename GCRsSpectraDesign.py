# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '~/QTDes/GCRs Spectra/GCRsSpectraDesign.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(778, 520)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(778, 519))
        MainWindow.setMaximumSize(QtCore.QSize(778, 520))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(32, 74, 135))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(48, 111, 203))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(40, 92, 169))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(16, 37, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 49, 90))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(143, 164, 195))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(32, 74, 135))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(48, 111, 203))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(40, 92, 169))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(16, 37, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 49, 90))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(143, 164, 195))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(16, 37, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(32, 74, 135))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(48, 111, 203))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(40, 92, 169))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(16, 37, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 49, 90))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(16, 37, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(16, 37, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(32, 74, 135))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        MainWindow.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(14)
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        MainWindow.setFont(font)
        MainWindow.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Icon.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setLayoutDirection(QtCore.Qt.RightToLeft)
        MainWindow.setIconSize(QtCore.QSize(48, 48))
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Bt_Calculate = QtWidgets.QPushButton(self.centralwidget)
        self.Bt_Calculate.setGeometry(QtCore.QRect(540, 480, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Bt_Calculate.setFont(font)
        self.Bt_Calculate.setStatusTip("")
        self.Bt_Calculate.setCheckable(False)
        self.Bt_Calculate.setObjectName("Bt_Calculate")
        self.Slider_W = QtWidgets.QSlider(self.centralwidget)
        self.Slider_W.setGeometry(QtCore.QRect(10, 170, 391, 31))
        self.Slider_W.setStatusTip("")
        self.Slider_W.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Slider_W.setMaximum(300)
        self.Slider_W.setProperty("value", 0)
        self.Slider_W.setSliderPosition(0)
        self.Slider_W.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_W.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.Slider_W.setTickInterval(10)
        self.Slider_W.setObjectName("Slider_W")
        self.Ion_Ch = QtWidgets.QComboBox(self.centralwidget)
        self.Ion_Ch.setGeometry(QtCore.QRect(420, 140, 351, 41))
        self.Ion_Ch.setAcceptDrops(True)
        self.Ion_Ch.setStatusTip("")
        self.Ion_Ch.setWhatsThis("")
        self.Ion_Ch.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Ion_Ch.setEditable(False)
        self.Ion_Ch.setMaxVisibleItems(28)
        self.Ion_Ch.setMinimumContentsLength(1)
        self.Ion_Ch.setFrame(True)
        self.Ion_Ch.setObjectName("Ion_Ch")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Ion_Ch.addItem("")
        self.Label_Emin = QtWidgets.QLabel(self.centralwidget)
        self.Label_Emin.setGeometry(QtCore.QRect(410, 220, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.Label_Emin.setFont(font)
        self.Label_Emin.setStatusTip("")
        self.Label_Emin.setObjectName("Label_Emin")
        self.Label_Emax = QtWidgets.QLabel(self.centralwidget)
        self.Label_Emax.setGeometry(QtCore.QRect(640, 220, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.Label_Emax.setFont(font)
        self.Label_Emax.setStatusTip("")
        self.Label_Emax.setObjectName("Label_Emax")
        self.EminVal = QtWidgets.QLineEdit(self.centralwidget)
        self.EminVal.setGeometry(QtCore.QRect(480, 200, 31, 25))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.EminVal.setFont(font)
        self.EminVal.setStatusTip("")
        self.EminVal.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.EminVal.setObjectName("EminVal")
        self.EmaxVal = QtWidgets.QLineEdit(self.centralwidget)
        self.EmaxVal.setGeometry(QtCore.QRect(710, 200, 31, 25))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.EmaxVal.setFont(font)
        self.EmaxVal.setStatusTip("")
        self.EmaxVal.setObjectName("EmaxVal")
        self.Line = QtWidgets.QFrame(self.centralwidget)
        self.Line.setGeometry(QtCore.QRect(-30, 110, 811, 20))
        self.Line.setFrameShape(QtWidgets.QFrame.HLine)
        self.Line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Line.setObjectName("Line")
        self.Label_WolfCap = QtWidgets.QLabel(self.centralwidget)
        self.Label_WolfCap.setGeometry(QtCore.QRect(80, 140, 181, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Label_WolfCap.setFont(font)
        self.Label_WolfCap.setToolTip("")
        self.Label_WolfCap.setObjectName("Label_WolfCap")
        self.Label_WVal = QtWidgets.QLabel(self.centralwidget)
        self.Label_WVal.setGeometry(QtCore.QRect(300, 140, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Label_WVal.setFont(font)
        self.Label_WVal.setObjectName("Label_WVal")
        self.Label_BinN = QtWidgets.QLabel(self.centralwidget)
        self.Label_BinN.setGeometry(QtCore.QRect(410, 260, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.Label_BinN.setFont(font)
        self.Label_BinN.setObjectName("Label_BinN")
        self.BinNVal = QtWidgets.QLineEdit(self.centralwidget)
        self.BinNVal.setGeometry(QtCore.QRect(620, 260, 141, 31))
        self.BinNVal.setStatusTip("")
        self.BinNVal.setObjectName("BinNVal")
        self.Ch_FE_dE = QtWidgets.QCheckBox(self.centralwidget)
        self.Ch_FE_dE.setGeometry(QtCore.QRect(750, 410, 20, 23))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.Ch_FE_dE.setFont(font)
        self.Ch_FE_dE.setStatusTip("")
        self.Ch_FE_dE.setText("")
        self.Ch_FE_dE.setObjectName("Ch_FE_dE")
        self.Ch_FE = QtWidgets.QCheckBox(self.centralwidget)
        self.Ch_FE.setGeometry(QtCore.QRect(750, 380, 20, 23))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.Ch_FE.setFont(font)
        self.Ch_FE.setStatusTip("")
        self.Ch_FE.setText("")
        self.Ch_FE.setObjectName("Ch_FE")
        self.Ch_TotalF = QtWidgets.QCheckBox(self.centralwidget)
        self.Ch_TotalF.setGeometry(QtCore.QRect(750, 440, 20, 23))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.Ch_TotalF.setFont(font)
        self.Ch_TotalF.setStatusTip("")
        self.Ch_TotalF.setText("")
        self.Ch_TotalF.setCheckable(True)
        self.Ch_TotalF.setChecked(False)
        self.Ch_TotalF.setObjectName("Ch_TotalF")
        self.Line_2 = QtWidgets.QFrame(self.centralwidget)
        self.Line_2.setGeometry(QtCore.QRect(400, 300, 381, 20))
        self.Line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.Line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Line_2.setObjectName("Line_2")
        self.Bt_Plot = QtWidgets.QPushButton(self.centralwidget)
        self.Bt_Plot.setGeometry(QtCore.QRect(679, 480, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.Bt_Plot.setFont(font)
        self.Bt_Plot.setStatusTip("")
        self.Bt_Plot.setObjectName("Bt_Plot")
        self.Bt_Clear = QtWidgets.QPushButton(self.centralwidget)
        self.Bt_Clear.setGeometry(QtCore.QRect(410, 480, 121, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.Bt_Clear.setFont(font)
        self.Bt_Clear.setStatusTip("")
        self.Bt_Clear.setCheckable(False)
        self.Bt_Clear.setObjectName("Bt_Clear")
        self.History = QtWidgets.QTextEdit(self.centralwidget)
        self.History.setGeometry(QtCore.QRect(10, 210, 391, 301))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.History.setFont(font)
        self.History.setObjectName("History")
        self.Label_WolfCap_2 = QtWidgets.QLabel(self.centralwidget)
        self.Label_WolfCap_2.setGeometry(QtCore.QRect(10, 0, 601, 51))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Label_WolfCap_2.setFont(font)
        self.Label_WolfCap_2.setToolTip("")
        self.Label_WolfCap_2.setAlignment(QtCore.Qt.AlignJustify|QtCore.Qt.AlignVCenter)
        self.Label_WolfCap_2.setWordWrap(True)
        self.Label_WolfCap_2.setObjectName("Label_WolfCap_2")
        self.Label_WolfCap_3 = QtWidgets.QLabel(self.centralwidget)
        self.Label_WolfCap_3.setGeometry(QtCore.QRect(100, 50, 511, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(True)
        font.setUnderline(True)
        font.setWeight(50)
        self.Label_WolfCap_3.setFont(font)
        self.Label_WolfCap_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTop|QtCore.Qt.AlignTrailing)
        self.Label_WolfCap_3.setWordWrap(True)
        self.Label_WolfCap_3.setOpenExternalLinks(True)
        self.Label_WolfCap_3.setObjectName("Label_WolfCap_3")
        self.Label_WolfCap_4 = QtWidgets.QLabel(self.centralwidget)
        self.Label_WolfCap_4.setGeometry(QtCore.QRect(410, 320, 241, 20))
        self.Label_WolfCap_4.setStatusTip("")
        self.Label_WolfCap_4.setObjectName("Label_WolfCap_4")
        self.Ch_FE_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.Ch_FE_2.setGeometry(QtCore.QRect(750, 350, 20, 23))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.Ch_FE_2.setFont(font)
        self.Ch_FE_2.setStatusTip("")
        self.Ch_FE_2.setText("")
        self.Ch_FE_2.setObjectName("Ch_FE_2")
        self.Logo = QtWidgets.QLabel(self.centralwidget)
        self.Logo.setGeometry(QtCore.QRect(628, 10, 141, 101))
        self.Logo.setText("")
        self.Logo.setObjectName("Logo")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(610, 0, 20, 121))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.Ch_FE_2_txt = QtWidgets.QLabel(self.centralwidget)
        self.Ch_FE_2_txt.setGeometry(QtCore.QRect(475, 352, 270, 17))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Ch_FE_2_txt.setFont(font)
        self.Ch_FE_2_txt.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.Ch_FE_2_txt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Ch_FE_2_txt.setObjectName("Ch_FE_2_txt")
        self.Ch_FE_txt = QtWidgets.QLabel(self.centralwidget)
        self.Ch_FE_txt.setGeometry(QtCore.QRect(475, 382, 270, 17))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.Ch_FE_txt.setFont(font)
        self.Ch_FE_txt.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.Ch_FE_txt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Ch_FE_txt.setObjectName("Ch_FE_txt")
        self.Ch_FE_dE_txt = QtWidgets.QLabel(self.centralwidget)
        self.Ch_FE_dE_txt.setGeometry(QtCore.QRect(475, 410, 270, 20))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.Ch_FE_dE_txt.setFont(font)
        self.Ch_FE_dE_txt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Ch_FE_dE_txt.setObjectName("Ch_FE_dE_txt")
        self.Ch_TotalF_txt = QtWidgets.QLabel(self.centralwidget)
        self.Ch_TotalF_txt.setGeometry(QtCore.QRect(435, 440, 311, 20))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.Ch_TotalF_txt.setFont(font)
        self.Ch_TotalF_txt.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Ch_TotalF_txt.setObjectName("Ch_TotalF_txt")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.Ion_Ch.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "<GN> GCRs Spectra"))
        self.Bt_Calculate.setToolTip(_translate("MainWindow", "Calculate spectra and generate selected ASCII files"))
        self.Bt_Calculate.setText(_translate("MainWindow", "Calculate"))
        self.Slider_W.setToolTip(_translate("MainWindow", "Choose the Wolf number value"))
        self.Ion_Ch.setToolTip(_translate("MainWindow", "Choose required GCR particle (if left as is will be calculated for all of the Ions)"))
        self.Ion_Ch.setCurrentText(_translate("MainWindow", "Choose the GCR particle"))
        self.Ion_Ch.setItemText(0, _translate("MainWindow", "Choose the GCR particle"))
        self.Ion_Ch.setItemText(1, _translate("MainWindow", "Z = 1 (H)"))
        self.Ion_Ch.setItemText(2, _translate("MainWindow", "Z = 2 (He)"))
        self.Ion_Ch.setItemText(3, _translate("MainWindow", "Z = 3 (Li)"))
        self.Ion_Ch.setItemText(4, _translate("MainWindow", "Z = 4 (Be)"))
        self.Ion_Ch.setItemText(5, _translate("MainWindow", "Z = 5 (B)"))
        self.Ion_Ch.setItemText(6, _translate("MainWindow", "Z = 6 (C)"))
        self.Ion_Ch.setItemText(7, _translate("MainWindow", "Z = 7 (N)"))
        self.Ion_Ch.setItemText(8, _translate("MainWindow", "Z = 8 (O)"))
        self.Ion_Ch.setItemText(9, _translate("MainWindow", "Z = 9 (F)"))
        self.Ion_Ch.setItemText(10, _translate("MainWindow", "Z = 10 (Ne)"))
        self.Ion_Ch.setItemText(11, _translate("MainWindow", "Z = 11 (Na)"))
        self.Ion_Ch.setItemText(12, _translate("MainWindow", "Z = 12 (Mg)"))
        self.Ion_Ch.setItemText(13, _translate("MainWindow", "Z = 13 (Al)"))
        self.Ion_Ch.setItemText(14, _translate("MainWindow", "Z = 14 (Si)"))
        self.Ion_Ch.setItemText(15, _translate("MainWindow", "Z = 15 (P)"))
        self.Ion_Ch.setItemText(16, _translate("MainWindow", "Z = 16 (S)"))
        self.Ion_Ch.setItemText(17, _translate("MainWindow", "Z = 17 (Cl)"))
        self.Ion_Ch.setItemText(18, _translate("MainWindow", "Z = 18 (Ar)"))
        self.Ion_Ch.setItemText(19, _translate("MainWindow", "Z = 19 (K)"))
        self.Ion_Ch.setItemText(20, _translate("MainWindow", "Z = 20 (Ca)"))
        self.Ion_Ch.setItemText(21, _translate("MainWindow", "Z = 21 (Sc)"))
        self.Ion_Ch.setItemText(22, _translate("MainWindow", "Z = 22 (Ti)"))
        self.Ion_Ch.setItemText(23, _translate("MainWindow", "Z = 23 (V)"))
        self.Ion_Ch.setItemText(24, _translate("MainWindow", "Z = 24 (Cr)"))
        self.Ion_Ch.setItemText(25, _translate("MainWindow", "Z = 25 (Mn)"))
        self.Ion_Ch.setItemText(26, _translate("MainWindow", "Z = 26 (Fe)"))
        self.Ion_Ch.setItemText(27, _translate("MainWindow", "Z = 27 (Co)"))
        self.Ion_Ch.setItemText(28, _translate("MainWindow", "Z = 28 (Ni)"))
        self.Ion_Ch.setItemText(29, _translate("MainWindow", "All Ions"))
        self.Label_Emin.setToolTip(_translate("MainWindow", "Lower boundary of energy (in MeV/n!)"))
        self.Label_Emin.setText(_translate("MainWindow", "E<sub>min</sub>: 10"))
        self.Label_Emax.setToolTip(_translate("MainWindow", "Upper boundary of energy (in MeV/n!)"))
        self.Label_Emax.setText(_translate("MainWindow", "E<sub>max</sub>: 10"))
        self.EminVal.setToolTip(_translate("MainWindow", "Set the power of lower energy boundary"))
        self.EminVal.setText(_translate("MainWindow", "1"))
        self.EmaxVal.setToolTip(_translate("MainWindow", "Set the power of upper energy boundary"))
        self.EmaxVal.setText(_translate("MainWindow", "5"))
        self.Label_WolfCap.setText(_translate("MainWindow", "Wolf number value:"))
        self.Label_WVal.setToolTip(_translate("MainWindow", "Current Wolf number value"))
        self.Label_WVal.setText(_translate("MainWindow", "0"))
        self.Label_BinN.setToolTip(_translate("MainWindow", "Total number of energy bins"))
        self.Label_BinN.setText(_translate("MainWindow", "Number of bins:"))
        self.BinNVal.setToolTip(_translate("MainWindow", "Set total number of bins"))
        self.BinNVal.setText(_translate("MainWindow", "40"))
        self.Ch_FE_dE.setToolTip(_translate("MainWindow", "Save Fluence spectra to ASCII file"))
        self.Ch_FE.setToolTip(_translate("MainWindow", "Save Differential Fluence spectra to ASCII file"))
        self.Ch_TotalF.setToolTip(_translate("MainWindow", "Save Integral Fluence to ASCII file"))
        self.Bt_Plot.setToolTip(_translate("MainWindow", "Plot histograms using generated ASCII data"))
        self.Bt_Plot.setText(_translate("MainWindow", "Plot"))
        self.Bt_Clear.setToolTip(_translate("MainWindow", "Clear History window"))
        self.Bt_Clear.setText(_translate("MainWindow", "Clear History"))
        self.History.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.Label_WolfCap_2.setText(_translate("MainWindow", "The program for calculating spectra of GCR particles with spectrum modulation depending on the value of the Wolf number, the calculations are based on the article: "))
        self.Label_WolfCap_3.setToolTip(_translate("MainWindow", "Link to the model"))
        self.Label_WolfCap_3.setText(_translate("MainWindow", "<html><head/><body><p><a href=\"https://www.researchgate.net/publication/255700494_A_Ready-to-use_Galactic_Cosmic_Ray_Model\"><span style=\" font-family:\'Arial\'; text-decoration: underline; color:#000000;\">A ready-to-use galactic cosmic ray model (Daniel Matthiä, Thomas Berger, Alankrita I. Mrigakshi, Günther Reitz) Advances in Space Research 51 (2013) 329–338</span></a></p></body></html>"))
        self.Label_WolfCap_4.setToolTip(_translate("MainWindow", "Choose what to save to ASCII data file"))
        self.Label_WolfCap_4.setText(_translate("MainWindow", "Save to ASCII file:"))
        self.Ch_FE_2.setToolTip(_translate("MainWindow", "Save Double Differential Fluence spectra to ASCII file"))
        self.Logo.setToolTip(_translate("MainWindow", "GCRs Spectra by GN"))
        self.Ch_FE_2_txt.setText(_translate("MainWindow", "<html><head/><body><p>Double differential F(E), m<span style=\" vertical-align:super;\">-2</span>s<span style=\" vertical-align:super;\">-1</span>sr<span style=\" vertical-align:super;\">-1</span>(MeV/n)<span style=\" vertical-align:super;\">-1</span></p></body></html>"))
        self.Ch_FE_txt.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Differential F(E), m</span><span style=\" font-size:8pt; vertical-align:super;\">-2</span><span style=\" font-size:8pt;\">s</span><span style=\" font-size:8pt; vertical-align:super;\">-1</span><span style=\" font-size:8pt;\">(MeV/n)</span><span style=\" font-size:8pt; vertical-align:super;\">-1</span></p></body></html>"))
        self.Ch_FE_dE_txt.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">F(E)dE, m</span><span style=\" font-size:8pt; vertical-align:super;\">-2</span><span style=\" font-size:8pt;\">s</span><span style=\" font-size:8pt; vertical-align:super;\">-1</span></p></body></html>"))
        self.Ch_TotalF_txt.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt;\">Integral Fluence, m</span><span style=\" font-size:8pt; vertical-align:super;\">-2</span><span style=\" font-size:8pt;\">s</span><span style=\" font-size:8pt; vertical-align:super;\">-1</span></p></body></html>"))

