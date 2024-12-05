# from .linear_algebra import Array 
from .dydx import Scalar 
from typing import List # , Tuple
#from math import e,log2
# import random # import randint, uniform, seed 

# # taylor series of natural logaritm 
TERMS = 10 
def ln(x):
	res= []
	for i in range(1,TERMS,1):
		res.append( (1/i ) * ((x- 1)/x)**i)
	return sum(res)
			
	
def flatten(v):
    for i in v:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i



def accuracy(y,y_hat):
	y_hat = list(flatten(y_hat))
	y = list(flatten(y))
	y_prediction = [1. if x.data > 0.5 else 0.0 for x in y_hat]
	matches = [1 if y1.data == y2 else 0 for (y1,y2) in zip(y, y_prediction)]
#	print("accuracy: y_prediction", y_prediction)
	
#	print("accuracy: y ", y)
	return (1/len(matches))*sum(matches)




class Loss:
	def __init__(self, name='binary_cross_entropy'):
		self.name = name
		
	def binary_cross_entropy(self, y, y_hat):
		'''
		y_hat are normalised model's probability; binary (0,1)
		y are targets probabilities; single scalar value indicating class {0,1}
		'''
		#y_hat.values = [[sigmoid(v) for v in row] for row in y_hat.values] 

		#print('indide loss y_hat\n', y_hat.values[:10][:10])
		#print(' inside loss y\n', y.values[:10][:10])
		acc = accuracy(y.values, y_hat.values)
		y_y_hat = list(zip(y.values,y_hat.values))
		# print('zip(y,y_hat\n', y_y_hat[:10])

		bce = lambda y, y_hat: -1*(y * ln(y_hat[0]) + (1-y) * ln(1-y_hat[0])) 
		
		
		tot_loss = [bce(y,y_hat) for (y,y_hat) in y_y_hat ]
		#print(y_y_hat)
		avg_loss = (1/len(tot_loss)) * sum(tot_loss)
		return acc, avg_loss 

	def categorical_cross_entropy(y:List[List[Scalar]],y_hat:List[List[Scalar]]) -> List[List[Scalar]]:
		'''
		categorical cross entropy
		y_hat are normalised model's probability; multiclass vec
		y are targets probabilities 
		'''
		pass
		
	def __call__(self, y, y_hat):
		if self.name == 'binary_cross_entropy':
			return self.binary_cross_entropy(y,y_hat)
		if self.name == 'categorical_cross_entropy':
			return self.categorical_cross_entropy(y,y_hat)
			


		
		