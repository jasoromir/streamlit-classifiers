

import streamlit as st
import numpy as np 
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# import matplotlib.pyplot as plt
import time
from datetime import time, datetime
 

st.title('Streamlit project to visualize classification methods on different datasets')

# st.write("""
# 	# Instructions: 
# 	Choose one of the datasets and one classifier from the left menu.
# 	""")


# BLOCK TO SELECT AND VISUALIZE INFO ABOUT DATASETs
# Names to display in the dropdown menu
dataset_list = ['Iris', 'Breast Cancer', 'Wine', 'Diabetes']
# Corresponding attribute to load the data from sklearn.datasets
dataset_loaders = ['load_iris', 'load_breast_cancer', 'load_wine' , 'load_diabetes']

# Pick desired dataset to work with
dataset_name = st.sidebar.selectbox('Select dataset', dataset_list)

# Load data
def get_dataset(dataset_name):
	data = getattr(datasets, dataset_loaders[ dataset_list.index(dataset_name) ])()
	return data
data = get_dataset(dataset_name)

# Optional: Show description
dataset_info = data['DESCR'].split(':',2)
st.write('Selected dataset: ', dataset_info[1].split('*')[0].lstrip('\n'))
with st.beta_expander('Show Description'):
	st.write(dataset_info[2])

# Extract features and labels
X = data.data
Y = data.target



# BLOCK TO SELECT CLASSIFIER
classifier_list = ['KNN', 'SVM', 'Random Forest']
classifier_method = ['KNeighborsClassifier']
# Pick classifier from dropdown menu
classifier_name = st.sidebar.selectbox('Select classifier', classifier_list)

st.sidebar.write('Choose parameters for ',classifier_name, ' classifier')

def get_classifier(classifier_name):
	if classifier_name == 'KNN':
		K = st.sidebar.slider('K', 1,20,5)
		weights = st.sidebar.radio('Weights', options = ['uniform', 'distance'])
		clf = KNeighborsClassifier(K, weights = weights)
	elif classifier_name == 'SVM':
		C = st.sidebar.select_slider('C',options = [1,10,100,1000])
		kernel = st.sidebar.select_slider('kernel',options = ['linear', 'poly', 'rbf'])
		gamma = st.sidebar.select_slider('gamma',options = [10**(-1),10**(-2),10**(-3),10**(-4)]) 
		clf = SVC(C = C, kernel = kernel)

	return clf

clf = get_classifier(classifier_name)
st.write('')
st.write('### Selected Classifier: ', classifier_name)
st.write('Optional parameters:' )
	



st.write(classifier_name)








