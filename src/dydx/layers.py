from .linear_algebra import Array 
from .autodiff import Scalar 
#from typing import List, Tuple
#from math import e,log2
import random # import randint, uniform, seed 
from functools import reduce
import sys

# gather seed information 
seed = random.randint(1, sys.maxsize)
rdm = random.seed(seed)

# seed = 8433248431303025905 # 3786509482071647239 # _
# rdm = random.seed(seed)


print(f"layer init seed: {seed}")

class Layer:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    

class Embedding(Layer):
    def __init__(self, activation=True, dims=None, values=None):
        super().__init__()
        self.dims = dims
        if values is None:
            self._init_weights()
        else:
            self.wb= Array(values=values)
        self.seed = seed 
        self.activation = activation       
        
    def _init_weights(self):
        x= (6/sum(self.dims))**(1/2)
        self.wb =  Array( values = [ [Scalar(random.uniform(-x,x)) for _ in range(self.dims[1] )] for _ in range(self.dims[0])] )
    
    def forward(self,x):
        # expecting x to be index  of one-hot encoded vector 
        bd = x.dims[0]
        outs = []
        for idx in range(bd):

            out = Array([self.wb.values[int(x.values[idx][0].data)][:]])
            if self.activation:
                out.values = [[x.relu() for x in row] for row in out.values]
            outs.append(out)
            
        stack = lambda x, y : x.stack(y,dim=1)      
        outs = reduce(stack, outs)
        return outs



    def __call__(self,x):
        return self.forward(x)
    
    def parameters(self):
        return [p for row in self.wb.values for p in row] 
        



class Linear(Layer):
	def __init__(self, activation=True, random=False, dims=None, values=None):

		super().__init__()
		self.dims = dims
  
		if values is None:
			self._init_weights()
		# need to override random
		else:
			self.wb= Array(values=values)
		
		self.seed = seed 
		self.activation = activation			
		
	def forward(self,x):
		# need to stack 1 along the end of x 
		# to be able to introduce bias
		# assumimg batch_dim,feature_dim
		bd = x.dims[0]
		
		#self.dims = (self.dims[0] +1, self.dims[1])
		x_b =  x.stack(Array(values=[[Scalar(1)]] * bd ), dim=0)
		# [[xs],
		#  [b] ] 
		
		out = x_b @ self.wb
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
		return [p for row in self.wb.values for p in row] 	
######
	
	def _init_weights(self):
		# xavier init
		x= (6/sum(self.dims))**(1/2)
		self.wb =  Array( values = [ [Scalar(random.uniform(-x,x)) for _ in range(self.dims[1] )] for _ in range(self.dims[0] +1)] )
  
  
	def __call__(self, x):
		return self.forward(x)
	
	# def step(self):
	# 	self.values = [[x._step() for x in row] for row in self.values]
	# 	return self 

