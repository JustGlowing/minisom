



import numpy as np

from matplotlib import image as img
from matplotlib import pyplot as plt

from minisom import * 

# load image as pixel array
image = img.imread('Lenna_(test_image).png')

plt.figure()
plt.imshow(image)
    
#y, x, rgb
img_dim = image.shape
x_max = img_dim[1]
y_max = img_dim[0]

#x, y : int
kernel_size = (80, 80)
stride = (20, 20)

img_processed = np.asarray(image)


# sub_iter = 4
# iters = sub_iter * 3
# for k in range(iters): 
#     p = int(k / sub_iter)
#     ii = int(k % sub_iter)
    
#     # print('ii: ', ii, 'iters: ', iters)
#     # if iters > 2:
#     #     if ii != (iters-1):
#     #         x_shift = int((kernel_size[0]-stride[0])/(iters - 2))
#     #         y_shift = int((kernel_size[1]-stride[1])/(iters - 2))
#     #     else:
#     #         x_shift = y_shift = 0
#     # else:
#     #     if ii%2 != 0:
#     #         x_shift = kernel_size[0]-stride[0]
#     #         y_shift = kernel_size[1]-stride[1]
#     #     else:
#     #         x_shift = y_shift = 0
    
    
#     # if ii%2 != 0:
#     #     x_shift = kernel_size[0]-stride[0]
#     #     y_shift = kernel_size[1]-stride[1]
#     # else:
#     #     x_shift = y_shift = 0
    
    
#     if p == 0 or p == 2:
#         if ii%2 != 0:
#             x_shift = kernel_size[0]-stride[0]
#             y_shift = kernel_size[1]-stride[1]
#         else:
#             x_shift = y_shift = 0
            
#     else:
#         if ii != (sub_iter-1):
#             x_shift = int((ii+1)*(kernel_size[0]-stride[0])/sub_iter)
#             y_shift = int((ii+1)*(kernel_size[1]-stride[1])/sub_iter)
#         else:
#             x_shift = y_shift = 0
            
            

            
            
    
#     print('x_shift: ', x_shift, 'y_shift: ', y_shift)
        
#     count = 0
#     #x axis
#     for i in range(0, x_max, stride[0]):
#         #y axis
#         for j in range(0, y_max, stride[1]):
#             print ('iter: %d %d/%d' % (ii, count, (x_max/stride[0]+(x_max%stride[0]>0))*(y_max/stride[1]+(y_max%stride[1]>0))))
#             x = kernel_size[0]
#             l = i       + x_shift
#             r = i + x   + x_shift
#             if r >= x_max:
#                 r = -1
                
#             y = kernel_size[1]
#             t = j       + y_shift
#             b = j + y   + y_shift
#             if b >= y_max:
#                 b = -1
            
#             if p == 0:
#                 num_iteration = (sub_iter-ii)*100
#             elif p == 1:
#                 num_iteration = 50
#             else: #p == 2
#                 num_iteration = 20
            
#             # no shifting or kernel no out of boundary
#             if (x_shift == 0 and y_shift == 0) or (r<=x_max and b<=y_max):
#                 img_slice = img_processed[l:r,t:b,:]
#                 x = img_slice.shape[0]
#                 y = img_slice.shape[1]
                
#                 som = MiniSom(x, y, 3, sigma=3., learning_rate=2.5, 
#                       neighborhood_function='gaussian', weights = img_slice)
                
#                 #flatten img slice as input for som
#                 som_input = img_slice.reshape((-1, 3))
#                 som.train(som_input, num_iteration, random_order=True, verbose=True)
                
#                 processed_slice = abs(som.get_weights())
#                 img_processed[l:r,t:b,:] = processed_slice
#             else:
#                 print('skip')
#             count += 1
#     #print after each iter
#     plt.figure()
#     plt.imshow(img_processed)
        




sub_iter = 2
iters = sub_iter * 4
for k in range(iters): 
    p = int(k / sub_iter)
    ii = int(k % sub_iter)
    
    # print('ii: ', ii, 'iters: ', iters)
    # if iters > 2:
    #     if ii != (iters-1):
    #         x_shift = int((kernel_size[0]-stride[0])/(iters - 2))
    #         y_shift = int((kernel_size[1]-stride[1])/(iters - 2))
    #     else:
    #         x_shift = y_shift = 0
    # else:
    #     if ii%2 != 0:
    #         x_shift = kernel_size[0]-stride[0]
    #         y_shift = kernel_size[1]-stride[1]
    #     else:
    #         x_shift = y_shift = 0
    
    
    # if ii%2 != 0:
    #     x_shift = kernel_size[0]-stride[0]
    #     y_shift = kernel_size[1]-stride[1]
    # else:
    #     x_shift = y_shift = 0
    
    

    if ii%2 != 0:
        x_shift = kernel_size[0]-stride[0]
        y_shift = kernel_size[1]-stride[1]
    else:
        x_shift = y_shift = 0
            

            

            
            
    
    print('x_shift: ', x_shift, 'y_shift: ', y_shift)
        
    count = 0
    #x axis
    for i in range(-x_shift, x_max, stride[0]):
        #y axis
        for j in range(y_shift, y_max, stride[1]):
            print ('iter: %d %d/%d' % (ii, count, (x_max/stride[0]+(x_max%stride[0]>0))*(y_max/stride[1]+(y_max%stride[1]>0))))
            
            
            x = kernel_size[0]
            l = i       
            r = i + x   
            if r >= x_max:
                r = -1
            if l < 0:
                l = 0
                
            y = kernel_size[1]
            t = j       
            b = j + y
            if b >= y_max:
                b = -1
            if t < 0:
                t = 0
            
            if p == 0:
                num_iteration = 40 #(sub_iter-ii)*50
            elif p == 1:
                num_iteration = 20
            elif p == 2: #p == 2
                num_iteration = 10
            else:
                num_iteration = 5
            
            img_slice = img_processed[l:r,t:b,:]
            x = img_slice.shape[0]
            y = img_slice.shape[1]
            
            som = MiniSom(x, y, 3, sigma=3., learning_rate=2.5, 
                  neighborhood_function='gaussian', weights = img_slice)
            
            #flatten img slice as input for som
            som_input = img_slice.reshape((-1, 3))
            som.train(som_input, num_iteration, random_order=True, verbose=True)
            
            processed_slice = abs(som.get_weights())
            img_processed[l:r,t:b,:] = processed_slice
            
            count += 1
            
    if ii == (sub_iter - 1):
        #print after each iter
        plt.figure()
        plt.imshow(img_processed)



