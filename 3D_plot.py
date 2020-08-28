# code for ploting is based on
# https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def cuboid_data(o, size=(1,1,1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    X *= 0.25
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"] * len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)] * len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data(p, size=s) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6, axis=0), **kwargs)

def plot_weights(weights):
    #transparency
    a = 1.0
    positions = np.zeros((N1*N2*N3, 3))
    plt_colors = np.zeros((N1*N2*N3, 4))
    
    
    idx = 0
    for i, x in enumerate(weights):
      for j, y in enumerate(x):
        for k, z in enumerate(y):
          p = [i, j, k]         #position
          c = np.append(z, [a]) #color
          positions[idx] = p
          plt_colors[idx] = c
          idx += 1
    
    
    
    ma = np.random.choice([0,1], size=(N1,N2,N3), p=[0.99, 0.01])
    x,y,z = np.indices((N1,N2,N3))-.5
    #positions = np.c_[x[ma==1],y[ma==1],z[ma==1]]
    #colors= np.random.rand(len(positions),4)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    
    pc = plotCubeAt(positions, colors=plt_colors,edgecolor="#00000020")
    ax.add_collection3d(pc)
    
    ax.set_xlim([0,N1])
    ax.set_ylim([0,N2])
    ax.set_zlim([0,N3])
    #plotMatrix(ax, ma)
    #ax.voxels(ma, edgecolor="k")
    
    plt.show()




#plot space x, y, z
N1 = N2 = N3 = 10


from minisom import * 

#Training inputs for RGBcolors
colors = [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]]

# colors = [[0., 0., 1.],
#       [0., 1., 0.],
#       [1., 0., 0.]]
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']
    
    
som = MiniSom(N1, N2, 3, sigma=3., 
              learning_rate=2.5, 
              neighborhood_function='gaussian', z = N3)
plot_weights(abs(som.get_weights()))

som.train(colors, 500, random_order=True, verbose=True)
plot_weights(abs(som.get_weights()))


def plot_weights(weights):
    #transparency
    a = 1.0
    positions = np.zeros((N1*N2*N3, 3))
    plt_colors = np.zeros((N1*N2*N3, 4))
    
    
    idx = 0
    for i, x in enumerate(weights):
      for j, y in enumerate(x):
        for k, z in enumerate(y):
          p = [i, j, k]         #position
          c = np.append(z, [a]) #color
          positions[idx] = p
          plt_colors[idx] = c
          idx += 1
    
    
    
    ma = np.random.choice([0,1], size=(N1,N2,N3), p=[0.99, 0.01])
    x,y,z = np.indices((N1,N2,N3))-.5
    #positions = np.c_[x[ma==1],y[ma==1],z[ma==1]]
    #colors= np.random.rand(len(positions),4)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_aspect('equal')
    
    pc = plotCubeAt(positions, colors=plt_colors,edgecolor="#00000020")
    ax.add_collection3d(pc)
    
    ax.set_xlim([0,N1])
    ax.set_ylim([0,N2])
    ax.set_zlim([0,N3])
    #plotMatrix(ax, ma)
    #ax.voxels(ma, edgecolor="k")
    
    plt.show()