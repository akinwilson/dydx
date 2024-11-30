from .linear_algebra import Array 
from .dydx import Scalar 
from typing import List, Tuple
#from math import e,log2
import random # import randint, uniform, seed 

import sys

# gather seed information 
seed = random.randint(1, sys.maxsize)
rdm = random.seed(seed)
print(f"layer init seed: {seed}")
# print(factorial(5))


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

class Layer:
	
    def _zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []



class Embedding(Array,Layer):
	def __init__(self, activation=True, random=False, dims=(32768,32), values=None):
		

		super().__init__(values=values,random=random, dims=dims)
		
		if values is None:
			self._init_weights()
		# need to override random
		else:
			self.values=values
		
		self.activation = activation			
		
	def forward(self,one_hot_encoding_idx):
		# need to stack 1 along the end of x 
		# to be able to introduce bias
		# assumimg batch_dim,feature_dim
		bd = x.dims[0]
		# print("dism before bias inclusion",x.dims)
		#self.dims = (self.dims[0] +1, self.dims[1])
		x_b =  x.stack(Array(values=[[Scalar(1)]] * bd ), dim=0)

		# print("dims after bias inclusion",x_b.dims)
		#xbdims = self.extract_dims(x_b.values, [])
		#selfdims = self.extract_dims(self.values, [])
		#print('x_b.dims from extract', xbdims)
		#print('self.dims from extract', selfdims)		
		# print("self dims", self.dims)
		out = x_b @ self
#		out = Array(values=out.values)
		return out
		
#####
	def parameters(self):
		return [p for row in self.values for p in row] 
#		return self.values		
######
	
	def _init_weights(self):
		# xavier init
		x= (6/sum(self.dims))**(1/2)
		self.values =  [ [Scalar(random.uniform(-x,x)) for _ in range(self.dims[1] )] for _ in range(self.dims[0] +1)]
		self.dims = self.extract_dims(self.values, [])
	
	 
	def __call__(self, x):
		return self.forward(x)





class Linear(Array,Layer):
	def __init__(self, activation=True, random=False, dims=None, values=None):
		

		super().__init__(values=values,random=random, dims=dims)
		
		if values is None:
			self._init_weights()
		# need to override random
		else:
			self.values=values
		
		self.activation = activation			
		
	def forward(self,x):
		# need to stack 1 along the end of x 
		# to be able to introduce bias
		# assumimg batch_dim,feature_dim
		bd = x.dims[0]
		
		#self.dims = (self.dims[0] +1, self.dims[1])
		x_b =  x.stack(Array(values=[[Scalar(1)]] * bd ), dim=0)
		#xbdims = self.extract_dims(x_b.values, [])
		#selfdims = self.extract_dims(self.values, [])
		#print('x_b.dims from extract', xbdims)
		#print('self.dims from extract', selfdims)		
		
		out = x_b @ self
#		out = Array(values=out.values)
		if self.activation:
			out.values = [[x.relu() for x in row] for row in out.values] 
			# out.values, lambda x: x.relu())
		else:
		#	self.apply(out.values, lambda x: x.sigmoid())
			out.values = [[x.sigmoid() for x in row] for row in out.values]
		return out
		
#####
	def parameters(self):
		return [p for row in self.values for p in row] 
#		return self.values		
######
	
	def _init_weights(self):
		# xavier init
		x= (6/sum(self.dims))**(1/2)
		self.values =  [ [Scalar(random.uniform(-x,x)) for _ in range(self.dims[1] )] for _ in range(self.dims[0] +1)]
		self.dims = self.extract_dims(self.values, [])

	def apply(self, item, fun):
		if isinstance(item, list):
			return [self.apply(x, fun) for x in item]
		else:
		    return fun(item) 	
	 
	def __call__(self, x):
		return self.forward(x)


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
			


		
		