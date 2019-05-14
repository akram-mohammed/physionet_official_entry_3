#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import os, shutil, zipfile
from numpy import array
import csv
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import entropy
import scipy as sc
from zipfile import ZipFile
import joblib

def load_sepsis_model():
	# Load the saved model pickle file
	Trained_model = joblib.load('saved_model.pkl')
	return Trained_model

def get_sepsis_score(data1, Trained_model):
	#Testing
	t=1
	df_test = np.array([], dtype=np.float64)
	df_test1 = pd.DataFrame()
	l = len(data1)
	
	df_test = data1
	df_test1 = pd.DataFrame(df_test)
	df_test2 = df_test1

	df_test2.columns = ['HR','O2Sat','Temp','SBP','MAP','DBP','Resp','EtCO2','BaseExcess','HCO3','FiO2','pH','PaCO2','SaO2','AST','BUN','Alkalinephos','Calcium','Chloride','Creatinine','Bilirubin_direct','Glucose','Lactate','Magnesium','Phosphate','Potassium','Bilirubin_total','TroponinI','Hct','Hgb','PTT','WBC','Fibrinogen','Platelets','Age','Gender','Unit1','Unit2','HospAdmTime','ICULOS']

	#Forward fill missing values
	df_test2.fillna(method='ffill', axis=0, inplace=True)
	df_test3 = df_test2.fillna(0)
	df_test = df_test3
	
	df_test['ID'] = 0
	DBP = pd.pivot_table(df_test,values='DBP',index='ID',columns='ICULOS')
	O2Sat = pd.pivot_table(df_test,values='O2Sat',index='ID',columns='ICULOS')
	Temp = pd.pivot_table(df_test,values='Temp',index='ID',columns='ICULOS')
	RR = pd.pivot_table(df_test,values='Resp',index='ID',columns='ICULOS')
	BP = pd.pivot_table(df_test,values='SBP',index='ID',columns='ICULOS')
	latest = pd.pivot_table(df_test,values='HR',index='ID',columns='ICULOS')

	Heart_rate_test = latest 
	RR_test = RR 
	BP_test = BP 
	DBP_test = DBP 
	Temp_test = Temp 
	O2Sat_test = O2Sat 

	result = Heart_rate_test

	result = result.fillna(0)
	RR_test = RR_test.fillna(0)
	BP_test = BP_test.fillna(0)
	Temp_test = Temp_test.fillna(0)
	DBP_test = DBP_test.fillna(0)
	O2Sat_test = O2Sat_test.fillna(0)
	
	#Since we are using a windows-based approach (6-hour window size), we pad our output for the 6 hours following patients admission.

	scores1 = 0
	labels1 = 0
	
	if l <7:
		scores1=1
		labels1=1
	else:
		#Get dataframe of probs
		#Windows based approach
		Heart_rate_test = result.iloc[:, l-6:l]
		RR2_test = RR_test.iloc[:, l-6:l]
		BP2_test = BP_test.iloc[:, l-6:l]
		Temp2_test = Temp_test.iloc[:, l-6:l]
		DBP2_test = DBP_test.iloc[:, l-6:l]
		O2Sat2_test = O2Sat_test.iloc[:, l-6:l]

		result['HR_min'] = Heart_rate_test.min(axis=1)
		result['HR_mean'] = Heart_rate_test.mean(axis=1)
		result['HR_max'] = Heart_rate_test.max(axis=1)
		result['HR_stdev'] = Heart_rate_test.std(axis=1)
		result['HR_var'] = Heart_rate_test.var(axis=1)
		result['HR_skew'] = Heart_rate_test.skew(axis=1)
		result['HR_kurt'] = Heart_rate_test.kurt(axis=1)
		
		result['BP_min'] = BP2_test.min(axis=1)
		result['BP_mean'] = BP2_test.mean(axis=1)
		result['BP_max'] = BP2_test.max(axis=1)
		result['BP_stdev'] = BP2_test.std(axis=1)
		result['BP_var'] = BP2_test.var(axis=1)
		result['BP_skew'] = BP2_test.skew(axis=1)
		result['BP_kurt'] = BP2_test.kurt(axis=1)

		result['RR_min'] = RR2_test.min(axis=1)
		result['RR_mean'] = RR2_test.mean(axis=1)
		result['RR_max'] = RR2_test.max(axis=1)
		result['RR_stdev'] = RR2_test.std(axis=1)
		result['RR_var'] = RR2_test.var(axis=1)
		result['RR_skew'] = RR2_test.skew(axis=1)
		result['RR_kurt'] = RR2_test.kurt(axis=1)

		result['DBP_min'] = DBP2_test.min(axis=1)
		result['DBP_mean'] = DBP2_test.mean(axis=1)
		result['DBP_max'] = DBP2_test.max(axis=1)
		result['DBP_stdev'] = DBP2_test.std(axis=1)
		result['DBP_var'] = DBP2_test.var(axis=1)
		result['DBP_skew'] = DBP2_test.skew(axis=1)
		result['DBP_kurt'] = DBP2_test.kurt(axis=1)

		result['O2Sat_min'] = O2Sat2_test.min(axis=1)
		result['O2Sat_mean'] = O2Sat2_test.mean(axis=1)
		result['O2Sat_max'] = O2Sat2_test.max(axis=1)
		result['O2Sat_stdev'] = O2Sat2_test.std(axis=1)
		result['O2Sat_var'] = O2Sat2_test.var(axis=1)
		result['O2Sat_skew'] = O2Sat2_test.skew(axis=1)
		result['O2Sat_kurt'] = O2Sat2_test.kurt(axis=1)

		result['Temp_min'] = Temp2_test.min(axis=1)
		result['Temp_mean'] = Temp2_test.mean(axis=1)
		result['Temp_max'] = Temp2_test.max(axis=1)
		result['Temp_stdev'] = Temp2_test.std(axis=1)
		result['Temp_var'] = Temp2_test.var(axis=1)
		result['Temp_skew'] = Temp2_test.skew(axis=1)
		result['Temp_kurt'] = Temp2_test.kurt(axis=1)
 
		X_test = result.values[:, Temp2_test.shape[1]:Temp2_test.shape[1]+42] 
		scores = Trained_model.predict_proba(X_test)
		scores1 = scores[0][1]

		if scores1>=0.3:
			labels1 = 1
		else:
			labels1 = 0
					
	return (scores1, labels1)