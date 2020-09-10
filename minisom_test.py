from minisom import * 

import numpy as np
import matplotlib.pyplot as plt

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

N1 = 10
N2 = 20
N3 = 5

som = MiniSom(N1, N2, 3, sigma=3., 
              learning_rate=2, 
              neighborhood_function='gaussian', random_seed = 1337)

plt.figure()
plt.imshow(abs(som.get_weights()), interpolation='none')

som.train(colors, 1000, random_order=True, verbose=True)

x = abs(som.get_weights())
# xmax, xmin = x.max(), x.min()
# x = (x - xmin)/(xmax - xmin)

plt.figure()
plt.imshow(x, interpolation='none')