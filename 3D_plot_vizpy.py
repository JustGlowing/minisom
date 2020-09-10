import numpy as np
import sys

from vispy import app, visuals, scene

from minisom import *


#get pos and color
def get_pos_and_color(weights):
    positions = np.zeros((N1*N2*N3, 3))
    colors = np.zeros((N1*N2*N3, 3))
    idx = 0
    
    for i, x in enumerate(weights):
      for j, y in enumerate(x):
        for k, z in enumerate(y):
          p = [i, j, k]         #position
          c = z #color
          #c = np.append(z, [a]) #color
          positions[idx] = p
          colors[idx] = c
          
          idx += 1
    
    return positions, colors

def plot_pos_and_color(positions, colors):
    # build your visuals, that's all
    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 20
    view.camera.center = [N1/2, N2/2, N3/2]
    
    p1 = Scatter3D(parent=view.scene)
    p1.set_gl_state('translucent', blend=True, depth_test=True)
    p1.set_data(positions, face_color=colors, symbol='o', size=10)
    canvas.show()

#map space x, y, z
N1 = 10
N2 = 10
N3 = 10

from minisom import * 

#Training inputs for RGBcolors
data = [[0., 0., 0.],
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

#initialize som
som = MiniSom(N1, N2, 3, sigma=3., learning_rate=2.5, 
              neighborhood_function='gaussian', z = N3)
#get positions and colors, then plot
weights = abs(som.get_weights())
pos, color = get_pos_and_color(weights)
plot_pos_and_color(pos, color)

#train som
som.train(data, 500, random_order=True, verbose=True)
weights = abs(som.get_weights())
pos, color = get_pos_and_color(weights)
plot_pos_and_color(pos, color)




