# load the digits dataset from scikit-learn
# 901 samples, about 180 samples per class 
# the digits represented 0,1,2,3,4
from sklearn import datasets
digits = datasets.load_digits(n_class=4)
data = digits.data # matrix where each row is a vector that represent a digit.
num = digits.target # num[i] is the digit represented by data[i]

# training the som
from minisom import MiniSom
som = MiniSom(20,20,64,sigma=0.3,learning_rate=0.6)
print("Training...")
som.train_random(data,1500) # random training
print("\n...ready!")

# plotting the result
from pylab import text,show,cm,axis
for x,t in zip(data,num):
	w = som.winner(x) # getting the winner
	text(w[0]+.5, w[1]+.5, str(t), color=cm.Dark2(t / 5.),fontdict={'weight': 'bold', 'size': 11})
axis([0,som.weights.shape[0],0,som.weights.shape[1]])
show() # show the figure