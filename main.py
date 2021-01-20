

import streamlit as st
import numpy as np 
import pandas as pd
from sklearn import datasets
# import matplotlib.pyplot as plt
import time
from datetime import time, datetime
 

st.title('Streamlit project to visualize classification methods on different datasets')

color = st.color_picker('Pick A Color', '#00f900')
st.write('The current color is', color)

st.write("""
	# Instructions: 
	Choose one of the datasets from the left menu.
	""")


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

classifier_list = ['KNN', 'LDA', 'SVM', 'Random Forest']
# Pick classifier from dropdown menu
classifier_name = st.sidebar.selectbox('Select classifier', ('KNN', 'SVM', 'Random Forest'))

def get_classifier(classifier_name):
	pass









st.write(classifier_name)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [41.64, -0.88],
    columns=['lat', 'lon'])

st.map(map_data)







