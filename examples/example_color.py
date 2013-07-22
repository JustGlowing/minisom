from pylab import imread,imshow,figure,show,subplot,title
from numpy import reshape,flipud,unravel_index,zeros
from minisom import MiniSom

# read the image
img = imread('tree.jpg')

# reshaping the pixels matrix
pixels = reshape(img,(img.shape[0]*img.shape[1],3))

# SOM initialization and training
print('training...')
som = MiniSom(3,3,3,sigma=0.1,learning_rate=0.2) # 3x3 = 9 final colors
som.random_weights_init(pixels)
starting_weights = som.weights.copy() # saving the starting weights
som.train_random(pixels,100)

print('quantization...')
qnt = som.quantization(pixels) # quantize each pixels of the image
print('building new image...')
clustered = zeros(img.shape)
for i,q in enumerate(qnt): # place the quantized values into a new image
	clustered[unravel_index(i,dims=(img.shape[0],img.shape[1]))] = q
print('done.')

# show the result
figure(1)
subplot(221)
title('original')
imshow(flipud(img))
subplot(222)
title('result')
imshow(flipud(clustered))

subplot(223)
title('initial colors')
imshow(flipud(starting_weights),interpolation='none')
subplot(224)
title('learned colors')
imshow(flipud(som.weights),interpolation='none')

show()