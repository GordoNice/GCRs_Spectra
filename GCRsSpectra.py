#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Info in README.md

import sys  
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIntValidator
from PyQt5.QtGui import QPixmap

# Code for design
import GCRsSpectraDesign

# Constants
E0_MeV, E0_GeV = 938.279, 0.938279  # The rest energy of proton

# DataFrame with model parameters for all of the ions 
# Format --- <Nucleus> | Nucleus | Z_i |  A_i | C_i | gamma_i | alpha_i |
#                      | 0	     | 1   |   2  |  3  |   4     |    5    |
df_Ions = pd.DataFrame(
	[
		["H", 1, 1, 18.5, 2.74, 2.85],
		["He", 2, 4, 3.69, 2.77, 3.12],
		["Li", 3, 6.9, 0.0195, 2.82, 3.41],
		["Be", 4, 9, 0.0177, 3.05, 4.3],
		["B", 5, 10.8, 0.04920, 2.96, 3.93],
		["C", 6, 12, 0.103, 2.76, 3.18],
		["N", 7, 14, 0.0367, 2.89, 3.77],
		["O", 8, 16, 0.0874, 2.7, 3.11],
		["F", 9, 19, 0.00319, 2.82, 4.05],
		["Ne", 10, 20.2, 0.0164, 2.76, 3.11],
		["Na", 11, 23, 0.00443, 2.84, 3.14],
		["Mg", 12, 24.3, 0.01930, 2.7, 3.65],
		["Al", 13, 27, 0.00417, 2.77, 3.46],
		["Si", 14, 28.1, 0.0134, 2.66, 3.00],
		["P", 15, 31, 0.00115, 2.89, 4.04],
		["S", 16, 32.1, 0.00306, 2.71, 3.3],
		["Cl", 17, 35.4, 0.0013, 3.00, 4.40],
		["Ar", 18, 39.9, 0.00233, 2.93, 4.33],
		["K", 19, 39.1, 0.00187, 3.05, 4.49],
		["Ca", 20, 40.1, 0.00217, 2.77, 2.93],
		["Sc", 21, 44.9, 0.00074, 2.97, 3.78],
		["Ti", 22, 47.9, 0.00263, 2.99, 3.79],
		["V", 23, 50.9, 0.00123, 2.94, 3.5],
		["Cr", 24, 52, 0.00212, 2.89, 3.28],
		["Mn", 25, 54.9, 0.00114, 2.74, 3.29],
		["Fe", 26, 55.8, 0.00932, 2.63, 3.01],
		["Co", 27, 58.9, 0.0001, 2.63, 4.25],
		["Ni", 28, 58.7, 0.00048, 2.63, 3.52],
	])

df_Ions.columns = ['Nucleus', 'Z_i', 'A_i', 'C_i', 'gamma_i', 'alpha_i']


class GCRsSpectraApp(QtWidgets.QMainWindow, GCRsSpectraDesign.Ui_MainWindow):
	def __init__(self):
		
		super().__init__()
		self.setupUi(self)  # Design init

		logo_pixmap = QPixmap('Logo.png')  # Load Logo
		self.Logo.setPixmap(logo_pixmap)

		self.StartText()  # Print init info in the History window
	
		# Check if void
		self.EminVal.textChanged.connect(self.CheckValues)
		self.EmaxVal.textChanged.connect(self.CheckValues)
		self.BinNVal.textChanged.connect(self.CheckValues)

		# If slider position is changed than print W in the label
		self.Slider_W.valueChanged.connect(self.ChangeSlide)

		# If calculated button pressed init calculate function
		self.Bt_Calculate.clicked.connect(self.Calculate)
		# Draw plot if plot button is pressed
		self.Bt_Plot.clicked.connect(self.Plot)
		# Clear History window if clear button is clicked
		self.Bt_Clear.clicked.connect(self.StartText)
		self.Bt_Clear.clicked.connect(self.CheckPlot)

		# User can put only Int from this range in the fields
		self.EminVal.setValidator(QIntValidator(-6, 10))
		self.EmaxVal.setValidator(QIntValidator(-6, 10))
		self.BinNVal.setValidator(QIntValidator(10, 1000000))

		# Signal from CheckBox with Total Fluence rate
		self.Ch_TotalF.stateChanged.connect(self.CheckPlot)

		# Signal from ComboBox
		self.Ion_Ch.currentIndexChanged.connect(self.CheckPlot)

	def StartText(self):
		"""
		This function prints init info in the History window
		"""
		self.History.setText(' ')
		self.History.append('GCRs Spectra')
		self.History.append('Version 4.2' + '\n')
		self.History.append('Author: Gordeev Ivan <GN>')
		self.History.append('e-mail: gordeev@jinr.ru' + '\n')
		self.History.append('Dubna, Russia, 2019' + '\n')
		self.History.append('*********** History ***********\n\n')

	# Take value from slider in the label for W
	def ChangeSlide(self): 
		"""Takes W value from Slider position"""
		self.Label_WVal.setText(str(self.Slider_W.value()))

	def CheckPlot(self):
		if 0 < self.Ion_Ch.currentIndex() < 29 and self.Ch_TotalF.isChecked():
			self.Bt_Plot.setEnabled(False)

		if self.Ion_Ch.currentIndex() < 0 and self.Ch_FE_2.isChecked():
			self.Bt_Plot.setEnabled(False)

		if self.Ion_Ch.currentIndex() < 0 and self.Ch_FE.isChecked():
			self.Bt_Plot.setEnabled(False)

		if self.Ion_Ch.currentIndex() < 0 and self.Ch_FE_dE.isChecked():
			self.Bt_Plot.setEnabled(False)

		else:
			self.Bt_Plot.setEnabled(True)

	def CheckValues(self):
		"""This function checks input values for correctness and if they are 
		ok then makes Calculate and Plot buttons enabled"""
		if self.BinNVal.text() == '' or int(self.BinNVal.text()) < 10 \
					or self.EminVal.text() == '' or self.EmaxVal.text() == '' \
					or self.EminVal.text() >= self.EmaxVal.text():

			self.Bt_Calculate.setEnabled(False)
			self.Bt_Plot.setEnabled(False)
		else:
			self.Bt_Calculate.setEnabled(True)
			self.Bt_Plot.setEnabled(True)

	def InitConstants(self): 
		"""
		Took all the cons-ts from user input and return them as variables
		"""
		w = self.Slider_W.value()  # Wolf Number from slider

		emin = int(self.EminVal.text())  # Lower energy boundary
		emax = int(self.EmaxVal.text())  # Upper energy boundary
		bin_n = int(self.BinNVal.text())  # Number of bins

		ion_ch = int(self.Ion_Ch.currentIndex()) - 1  # Chosen ion

		return w, emin, emax, bin_n, ion_ch
	
	def Plot_Each(
			self, str_data, str_color, str_xlabel,
			str_ylabel, str_title, int_w, bool_flag):
		"""
		Function for plotting each of generated files
		"""
		if not bool_flag:
			data = np.genfromtxt(
				str_data, comments='#', delimiter='\t',
				names=['Emin', 'Emax', 'F'])
			
			plt.grid(True, color='gray', linestyle='-.', linewidth=0.6)

			plt.xlim(left=np.ndarray.min(data['Emin']))
			plt.xlim(right=np.ndarray.max(data['Emax']))

			plt.ylim(bottom=np.ndarray.min(data['F']))
			plt.ylim(top=np.ndarray.max(data['F']*1.2))

			plt.xscale('log')
			
			plt.fill_between(
				data['Emin'], data['F'], step="post", color=str_color, alpha=1.0)
			plt.fill_between(
				data['Emax'], data['F'], step="pre", color=str_color, alpha=1.0)

			plt.step(
				data['Emin'], data['F'],
				where='post', color=str_color, label=f'W = {int_w}')
			plt.step(data['Emax'], data['F'], where='pre', color=str_color)

			plt.xlabel(str_xlabel, style="normal")
			plt.ylabel(str_ylabel)

			plt.title(str_title)
			plt.legend()
			plt.show()

		else:
			data = np.genfromtxt(
				str_data, comments='#', delimiter='\t', names=['Z', 'Label', 'F'])
			
			plt.grid(True, color='gray', linestyle='dotted', linewidth=0.6)

			plt.xticks(np.arange(min(data['Z']), max(data['Z'])+1, 1.0))

			plt.plot(
				data['Z'], data['F'], color=str_color, marker='o', label=f'W = {int_w}')

			plt.xlabel(str_xlabel)
			plt.ylabel(str_ylabel)

			plt.title(str_title)
			plt.legend()
			plt.show()

	def Plot(self):
		"""Function for plotting """

		w, emin, emax, bin_n, ion_ch = self.InitConstants()

		# Standard Values for plots
		# Font sizes
		small_size = 12
		medium_size = small_size + 2
		bigger_size = medium_size + 4

		plt.rc('axes', titlesize=medium_size)  # Size of the axes headers
		plt.rc('axes', labelsize=medium_size)  # Font size of the x and y labels
		plt.rc('xtick', labelsize=small_size)  # Font size of the tick labels
		plt.rc('ytick', labelsize=small_size)  # Font size of the tick labels
		plt.rc('legend', fontsize=medium_size)  # Legend font size
		plt.rc('figure', titlesize=bigger_size)  # Font size of the figure title

		plt.rcParams['axes.formatter.min_exponent'] = 2
		plt.yscale('log')

		if self.Ch_FE_2.isChecked():
			self.Plot_Each(
				f'GCR_2DF(E)_{df_Ions.at[ion_ch,"Z_i"]}{df_Ions.at[ion_ch,"Nucleus"]}' +
				f'_W{w}_E{emin}_{emax}_binN{bin_n}.tsv', '#52B7BD',
				r'$Energy,\ MeV/n$', r'$F(E),\ (m^2 \cdot s \cdot sr \cdot MeV/n)^{-1}$',
				r'$Double\ differential\ fluence\ rate\ of\ particle\ with\ Z\ =\ $'
				+ str(df_Ions.at[ion_ch, 'Z_i']) + ' ('
				+ str(df_Ions.at[ion_ch, 'Nucleus']) + ')', w, False)

		if self.Ch_FE.isChecked():
			self.Plot_Each(
				f'GCR_DF(E)_{df_Ions.at[ion_ch,"Z_i"]}{df_Ions.at[ion_ch,"Nucleus"]}' +
				f'_W{w}_E{emin}_{emax}_binN{bin_n}.tsv', '#7fcdbb',
			 	r'$Energy,\ MeV/n$', r'$F(E),\ (m^2 \cdot s \cdot MeV/n)^{-1}$',
			  	r'$Differential\ fluence\ rate\ of\ particle\  with\ Z\ =\ $'
				+ str(df_Ions.at[ion_ch, 'Z_i']) + ' ('
				+ str(df_Ions.at[ion_ch, 'Nucleus']) + ')', w, False)

		if self.Ch_FE_dE.isChecked():
			self.Plot_Each(
				f'GCR_F(E)dE_{df_Ions.at[ion_ch,"Z_i"]}{df_Ions.at[ion_ch,"Nucleus"]}' +
				f'_W{w}_E{emin}_{emax}_binN{bin_n}.tsv', '#d95f0e',
				r'$Energy,\ MeV/n$', r'$F(E),\ m^{-2} \cdot s^{-1}$',
				r'$Fluence\ rate\ of\ particle\ with\ Z\ =\ $'
				+ str(df_Ions.at[ion_ch, 'Z_i']) + ' ('
				+ str(df_Ions.at[ion_ch, 'Nucleus']) + ')', w, False)

		if self.Ch_TotalF.isChecked():
			self.Plot_Each(
				f'GCR_IntegralF_All_W{w}_E{emin}_{emax}_binN{bin_n}.tsv',
				'green',
				r'$Nuclear\ Charge\ (Z)$',
				r'$Fluence\ rate,\ m^{-2}s^{-1}$',
				r'$Fluence\ rate\ of\ GCR\ nuclei$', w, True)

	def Calc_Plot(
			self, df_ebin, emin, emax, bin_n, df_ions,
			w, ion_ch, name_path, column, comment, col_cap):
		"""

		:param df_ebin:
		:type df_ebin: pd.DataFrame
		:param emin: minimum energy edge
		:type emin: int
		:param emax: maximum energy edge 
		:type emax: int
		:param bin_n: number of bins
		:type bin_n: int
		:param df_ions: DataFrame with ions
		:type df_ions: pd.DataFrame
		:param w: Wolf number
		:type w: int
		:param ion_ch: ion choice
		:type ion_ch: int
		:param name_path: 
		:type name_path: str
		:param column: 
		:type column: str
		:param comment: 
		:type comment: str
		:param col_cap: 
		:type col_cap: str
		"""
		path =\
			f'{name_path}_{df_ions.at[ion_ch, "Z_i"]}{df_ions.at[ion_ch, "Nucleus"]}'\
			+ f'_W{w}_E{emin}_{emax}_binN{bin_n}.tsv'
		df_ebin.to_csv(
			path, sep='\t',
			columns=['Emin', 'Emax', column], index=False, header=False)

		src = open(path, "r")

		fline =\
			f'{comment}{df_ions.at[ion_ch, "Z_i"]}\"{df_ions.at[ion_ch, "Nucleus"]}\"\n'\
			+ f'# Emin(MeV/n), Emax(MeV/n), {col_cap}\n'
		oline = src.readlines()
		oline.insert(0, fline)
		src.close()

		src = open(path, "w")
		src.writelines(oline)
		src.close()

	def Calc_df(
			self, df_ebin, emin, emax, bin_n, df_ions, w, ion_ch):
		"""
		Function makes specific dataframe for specific ion with parameters
		and returns this df as a result

		:param df_ebin: 
		:type df_ebin: 
		:param emin: 
		:type emin: int
		:param emax: 
		:type emax: int
		:param bin_n: 
		:type bin_n: int
		:param df_ions:
		:type df_ions: pd.DataFrame
		:param w: 
		:type w: int
		:param ion_ch: 
		:type ion_ch: int
		:return: 
		:rtype: 
		 """
		self.History.append("--------- Emin, MeV/n ---------")
		self.History.append(f'{df_ebin["Emin"]}\n')

		# Make lower boundary from upper by shifting
		df_ebin['Emax'] = df_ebin['Emin'].shift(-1)
		# And special treat for the final value
		df_ebin.at[bin_n-1, 'Emax'] = 10 ** emax

		self.History.append("--------- Emax, MeV/n ---------")
		self.History.append(f'{df_ebin["Emax"]}\n')

		self.History.append("--------- dE, MeV/n ---------")
		# Make energy intervals
		df_ebin['dE'] = df_ebin['Emax'] - df_ebin['Emin']
		self.History.append(f'{df_ebin["dE"]}\n')

		self.History.append("--------- beta ---------")
		df_ebin['beta'] =\
			np.sqrt(
				df_ebin['Emax']*(df_ebin['Emax']+2*E0_MeV))/(df_ebin['Emax']+E0_MeV)
		self.History.append(f'{df_ebin["beta"]}\n')

		self.History.append("--------- Rigidity, GV ---------")
		df_ebin['R'] = \
			(df_ions.at[ion_ch, 'A_i'] / df_ions.at[ion_ch, 'Z_i']) * np.sqrt(
				(df_ebin['Emax']/1e3)**2+2*(df_ebin['Emax']/1e3)*E0_GeV)
		self.History.append(f'{df_ebin["R"]}\n')

		self.History.append("--------- DDF, m-2s-1sr-1(MeV/n)-1 ---------")
		df_ebin['DDF'] =\
			(
					(
							df_ions.at[ion_ch, 'C_i']
							* df_ions.at[ion_ch, 'A_i']
							* df_ebin['beta'] ** df_ions.at[ion_ch, 'alpha_i']
					) /
					(
							df_ions.at[ion_ch, 'Z_i']
							* df_ebin['beta']
							* df_ebin['R'] ** df_ions.at[ion_ch, 'gamma_i']
					)
			)*(df_ebin['R']/(df_ebin['R']+(0.37+0.0003*w**1.45)))**(0.02*w+4.7)
		self.History.append(f'{df_ebin["DDF"]}\n')

		self.History.append("--------- DF, m-2s-1(MeV/n)-1 ---------")
		df_ebin['DF'] = 4*math.pi*df_ebin['DDF']
		self.History.append(f'{df_ebin["DF"]}\n')

		self.History.append("--------- F, m-2s-1 ---------")
		df_ebin['FdE'] = df_ebin['DF']*df_ebin['dE']
		self.History.append(f'{df_ebin["FdE"]}\n')

		if self.Ch_FE_2.isChecked():
			self.Calc_Plot(
				df_ebin, emin, emax, bin_n, df_ions, w, ion_ch,
				'GCR_2DF(E)', 'DDF',
				'# Double differential (by angle and energy) '
				+ 'fluence rate for particle with Z = ',
				'2DF(E)(m-2 s-1 sr-1 (MeV/n)-1)')
		if self.Ch_FE.isChecked():
			self.Calc_Plot(
				df_ebin, emin, emax, bin_n, df_ions, w, ion_ch,
				'GCR_DF(E)', 'DF',
				'# Differential (by energy) fluence rate for particle with Z = ',
				'DF(E)(m-2 s-1 (MeV/n)-1)')

		if self.Ch_FE_dE.isChecked():
			self.Calc_Plot(
				df_ebin, emin, emax, bin_n, df_ions, w, ion_ch,
				'GCR_F(E)dE', 'FdE',
				f'# Integral fluence rate = {df_ebin["FdE"].sum()} m-2 s-1.'
				+ ' Fluence rate for particle with Z = ', 'F(E)(m-2 s-1)')

		integral = df_ebin['FdE'].sum()
		self.History.append(f'Integral fluence rate = {integral} m-2 s-1\n')

		return df_ebin, integral

	def Calculate(self):
		"""
		Function provides all calculations and save data in some files
		"""
		w, emin, emax, bin_n, ion_ch = self.InitConstants()

		# Calculate 
		if emin > 0 and emax > 0:
			bin_delta = 1/(bin_n/(emax - emin))
		elif emin < 0 and emax < 0:
			bin_delta = 1/(bin_n/(math.fabs(emin) - math.fabs(emax)))
		else:
			bin_delta = 1/(bin_n/(math.fabs(emin) + math.fabs(emax)))

		df_ebin = pd.DataFrame(10**np.arange(emin, emax, bin_delta))
		df_ebin.columns = ['Emin']

		print(f'Chosen Ion Z = {ion_ch+1}\n')

		# Check if function for all spectra was chosen
		if ion_ch == 28 or ion_ch == -1:
			total_int = []
			for i in range(0, 28):
				df_res, integral = self.Calc_df(
					df_ebin, emin, emax, bin_n, df_Ions, w, i)
				total_int.append(integral)
				print(df_res, integral, sep='\n')

			df_total_int = pd.DataFrame(np.arange(1, 29), columns=['Z'])
			df_total_int['Ion'] = df_Ions['Nucleus']
			df_total_int['Integral fluence rate, m-2 s-1'] = total_int

			df_total_int.to_csv(
				f'GCR_IntegralF_All_W{w}_E{emin}_{emax}_binN{bin_n}.tsv',
				sep='\t', index=False, header=False)

			src = open(
				f'GCR_IntegralF_All_W{w}_E{emin}_{emax}_binN{bin_n}.tsv', "r")

			fline =\
				f'#Integral fluence rate for all particles at W = {w}'\
				+ '\n#Z	Ion	Integral fluence rate, m-2 s-1\n'
			oline = src.readlines()
			oline.insert(0, fline)
			src.close()

			src = open(
				f'GCR_IntegralF_All_W{w}_E{emin}_{emax}_binN{bin_n}.tsv', "w")
			src.writelines(oline)
			src.close()

			print(df_total_int)

		else:
			df_res, integral = self.Calc_df(
				df_ebin, emin, emax, bin_n, df_Ions, w, ion_ch)
			print(df_res)
			print(f'Integral fluence rate = {integral} m-2 s-1')


def main():
	app = QtWidgets.QApplication(sys.argv)  # New QApplication
	window = GCRsSpectraApp()  # Make object of the GCRsSpectraApp class
	window.show()  # Show this window
	app.exec_()  # Launch app


if __name__ == '__main__':
	main()
