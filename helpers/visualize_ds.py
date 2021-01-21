import streamlit as st
import numpy as np 
import pandas as pd
import altair as alt

def visualize(data):

	# Extract features and labels
	dataset_targets = data.target_names
	dataset_features = data.feature_names

	X = data.data
	Y = data.target


	# Optional: Show description
	dataset_info = data['DESCR'].split(':',2)
	st.write('Selected dataset: ', dataset_info[1].split('*')[0].lstrip('\n'))
	with st.beta_expander('Dataset Description'):
		st.write(dataset_info[2])

	# Optional visualize data
	with st.beta_expander('Dataset Visualization'):
		col1 , col2 = st.beta_columns(2)
		x_label = col1.selectbox('Select feature for X axis', dataset_features, index = 0)
		y_label = col2.selectbox('Select feature for Y axis', dataset_features, index = 1)
		c_label = dataset_targets
		
		try:
			x_data = X[:,dataset_features.index(x_label)]
			y_data = X[:,dataset_features.index(y_label)]
		except:
			x_data = X[:,dataset_features.tolist().index(x_label)]
			y_data = X[:,dataset_features.tolist().index(y_label)]
		c_data = dataset_targets[Y]

		

		df = pd.DataFrame(np.array([x_data, y_data, c_data]).T, columns = [x_label, y_label, 'classes'])

		a = alt.Chart(df).mark_area(opacity=0.8, interpolate='step'
			).encode(
			    alt.X(x_label+':Q', bin=alt.Bin(maxbins=25)),
			    alt.Y('count()', stack=None),
			    alt.Color('classes:N')
		    ).properties(height = 200)
		col1.altair_chart(a)
			
		b = alt.Chart(df).mark_area(opacity=0.8, interpolate='step'
			).encode(
			    alt.X(y_label+':Q', bin=alt.Bin(maxbins=25)),
			    alt.Y('count()', stack=None),
			    alt.Color('classes:N')
		    ).properties(height = 200)
		col2.altair_chart(b)

		c = alt.Chart(df).mark_circle(size = 60
			).encode(
				x = alt.X(x_label, axis = alt.Axis(labels = False)),
				y = alt.X(y_label, axis = alt.Axis(labels = False)), 
				color = 'classes' 
			).properties(width = 500, height = 300)
		st.altair_chart(c)