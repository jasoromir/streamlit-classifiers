

import streamlit as st
import numpy as np 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import plotly.express as px
import time
from helpers.visualize_ds import *



st.title('Streamlit for visualization and classification')

# BLOCK TO SELECT AND VISUALIZE INFO ABOUT DATASETs
# Names to display in the dropdown menu
dataset_list = ['Iris', 'Breast Cancer', 'Wine'] #, 'Diabetes'
# Corresponding attribute to load the data from sklearn.datasets
dataset_loaders = ['load_iris', 'load_breast_cancer', 'load_wine' ] #, 'load_diabetes'
# Pick desired dataset to work with
dataset_name = st.sidebar.selectbox('Select dataset', dataset_list)


# Load data
def get_dataset(dataset_name):
	data = getattr(datasets, dataset_loaders[ dataset_list.index(dataset_name) ])()
	return data
data = get_dataset(dataset_name)

# Helper function to visualize data
visualize(data)

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
	params = {}
	if classifier_name == 'KNN':
		K = st.sidebar.slider('K', 1,20,5)
		weights = st.sidebar.radio('Weights', options = ['uniform', 'distance'])
		clf = KNeighborsClassifier(K, weights = weights)
		params['K'] = K
		params['weights'] = weights
	elif classifier_name == 'SVM':
		# Pick a kernel
		kernel = st.sidebar.selectbox('kernel', ['linear', 'polynomial', 'rbf'])
		params['kernel'] = kernel
		# Pick the param C (i.e. low values->softer margin, high values-> hard margin)
		C = st.sidebar.select_slider('C',options = [0.001, 0.01, 0.1, 1,10,100,1000], value = 1)
		params['C'] = C	
		if kernel == 'polynomial':
			degree = st.sidebar.slider('degree (d)', 2,10,2,2)
			params['degree'] = degree
			clf = SVC(C = C, kernel = kernel, degree = degree)
		elif kernel == 'rbf':
			gamma = st.sidebar.select_slider('gamma',options = [10**(-1),10**(-2),10**(-3),10**(-4)]) 		
			params['gamma'] = gamma
			clf = SVC(C = C, kernel = kernel, gamma = gamma)
		else:
			clf = SVC(C = C, kernel = kernel)		

	return clf, params

clf, params = get_classifier(classifier_name)

with st.beta_expander('Dataset Classification'):
	st.write('### Selected Classifier: ', classifier_name)
	clf_info = 'Current parameters -> '
	for p_k, p_v in params.items():
		clf_info = clf_info.strip('\n') +'**' + p_k + '**: ' + str(p_v) + ', ' 
	st.write(clf_info )
	




mesh_size = .02
margin = 0.25

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(
    X[:,0:2], np.reshape(Y, (len(Y),1)), test_size=0.25, random_state=0)

# Create a mesh grid on which we will run our model
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

st.write(X_train.shape)
st.write(y_train.shape)
# Create classifier, run predictions on grid
clf.fit(X_train, y_train)
st.write(np.c_[xx.ravel(), yy.ravel()].shape)
Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)


# Plot the figure
fig = go.Figure(data=[
    go.Contour(
        x=xrange,
        y=yrange,
        z=Z,
        colorscale='RdBu',
        opacity=0.8,
        name='Score',
        hoverinfo='skip'
    )
])

st.write(fig)




