import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

st.sidebar.markdown('# MiniSom playground')

st.sidebar.markdown('Get MiniSom: https://github.com/JustGlowing/minisom')
st.sidebar.markdown('Follow MiniSom: https://twitter.com/JustGlowing')

@st.cache
def load_data():
	columns=['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel',
			 'asymmetry_coefficient', 'length_kernel_groove', 'target']
	data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', 
						names=columns, 
					    sep='\t+', engine='python')
	target = data['target'].values
	label_names = {1:'Kama', 2:'Rosa', 3:'Canadian'}
	data = data[data.columns[:-1]]
	# data normalization
	data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
	return data, target, label_names

data_load_state = st.text('Loading data...')
data_df, target, label_names = load_data()
data_load_state.text("Ready")

data = data_df.values

st.sidebar.markdown('## Parameters')
neighborhood_function = st.sidebar.selectbox('neighborhood_function', 
	                                 ['gaussian', 'mexican_hat', 'bubble', 'triangle'])
activation_distance = st.sidebar.selectbox('activation_distance', 
	                                 ['euclidean', 'cosine', 'manhattan', 'chebyshev'])
n_neurons = st.sidebar.slider('map size (N x N)', 1, 150, 9, step=1)
m_neurons = n_neurons
sigma = st.sidebar.slider('sigma', 0.1, 10.0, 1.5, step=0.1)
learning_rate = st.sidebar.slider('learning rate', 0.1, 50.0, 0.5, step=0.1)
iterations = st.sidebar.slider('training iterations', 0, 2000, 500, step=1)
som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=sigma,
              neighborhood_function=neighborhood_function, activation_distance=activation_distance,
              learning_rate=learning_rate, random_seed=0)

if st.sidebar.checkbox('PCA initialization'):
	som.pca_weights_init(data)
som.train(data, iterations, verbose=False)  # random training

fig = plt.figure(figsize=(10, 9))

markers = ['o', 's', 'D']
colors = ['C0', 'C1', 'C2']

single_marker = st.checkbox('Single Marker Visualization')
if single_marker:
	plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.5)  # plotting the distance map as background
	plt.colorbar()
	for cnt, xx in enumerate(data):
	    w = som.winner(xx)  # getting the winner
	    # palce a marker on the winning position for the sample xx
	    plt.plot(w[0]+.5, w[1]+.5, markers[target[cnt]-1], markerfacecolor='None',
	             markeredgecolor=colors[target[cnt]-1], markersize=12, markeredgewidth=2)
else:
	w_x, w_y = zip(*[som.winner(d) for d in data])
	w_x = np.array(w_x)
	w_y = np.array(w_y)

	plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
	plt.colorbar()

	for c in np.unique(target):
	    idx_target = target==c
	    plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
	                w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
	                s=20, c=colors[c-1], label=label_names[c])
	plt.legend(loc='upper right')

plt.tick_params(labelbottom=False, labelleft=False)
st.pyplot(fig)

st.write('`{e:2.6f}` Quantization error'.format(e=som.quantization_error(data)))
st.write('`{e:2.6f}` Topographic error'.format(e=som.topographic_error(data)))

"""
***
- The background represents the U-Matrix of the SOM (the darker, the more separated are the weights/codebooks).
- In the single marker visualization each marker represents a sample in the data but they're like to overlap.
- In the default visualization jittering is used to spread the markers in the cells.
- Each type of marker represents a class.
"""

st.sidebar.markdown('Dataset description: https://archive.ics.uci.edu/ml/datasets/seeds')

if st.checkbox('Show input data (before normalization)'):
	data_df['class'] = target
	data_df['class'] = data_df['class'].replace(label_names)
	st.write(data_df)
	del data_df['class']