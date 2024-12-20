# building block of computational graphs, the value 
from functools import reduce

product = lambda lst: reduce(lambda x,y: x*y, lst )
factorial = lambda N: product(list(range(1,N+1))) if N>=1 else 1

# no of terms in taylor series approx of exponential param 
TERMS = 5


def exp(x):
	res= []
	for i in range(0,TERMS):
		try:
			res.append( (1/factorial(i) ) * ((x)**i))
		except OverflowError:
			print(f" value causing overflow: {x}")
	return sum(res)

def sigmoid(x):
	# for numerical stability 
	return 1 / (1 + exp(-x))  if x > 0 else exp(x) / (1+exp(x))

class Scalar:
	
	def __init__(self,value,children=() ):
		self.data = value
		self.grad = 0
		self._backward = lambda : None
		self._previous = set(children)
		
 

	def __add__(self, other):
		other = other  if  isinstance(other, Scalar) else self.__class__(other)
		kwargs = {'value':self.data+other.data,
		                 'children':(self,other)} 
		result = self.__class__(**kwargs)
		
		def _backward():
			self.grad += result.grad
			other.grad += result.grad
		
		result._backward = _backward
		
		return result
		
		
	def __pow__(self, other):
		if  not isinstance(other, (int,float)):
			raise ValueError(f'Only supporting exponentiation with types int and float. received {type(other)}')
			
		kwargs = {'value':self.data**other,
		                 'children':(self,)} 
		result = self.__class__(**kwargs)
		
		def _backward():
			self.grad += (other*self.data**(other -1) )* result.grad
		
		result._backward = _backward
		
		return result

	def __repr__(self):
		return f'{self.data} | {self.grad}'
	
	def __mul__(self,other):
		other = other  if  isinstance(other, Scalar) else self.__class__(other)
		kwargs = {'value':self.data*other.data,
		                 'children':(self,other)} 
		result = self.__class__(**kwargs)
		
		def _backward():
			self.grad += other.data * result.grad
			other.grad += self.data * result.grad 
			
		result._backward = _backward
		
		return result



	def backward(self):
		seen = set()
		graph =[]
		def build_graph(node):
			if node not in seen:
				seen.add(node)
				for child in node._previous:
					build_graph(child)
			graph.append(node)
	    
		build_graph(self)
	    	
	    	# let the final output have partial 
	    	#derivative w.r.t itself be one 
		self.grad = 1
	    	# evaluate gradients in backwards orde
		for node in graph[::-1]:
			node._backward()
	    		
	    		
	
	def relu(self):
		kwargs = {'value':0 if self.data < 0 else self.data,
		                 'children':(self,)} 
		result = self.__class__(**kwargs)
		
		def _backward():
			self.grad += (result.data > 0 ) * result.grad
		
		result._backward = _backward
		return result

	
	def sigmoid(self):
		kwargs =  {'value':sigmoid(self.data) ,'children': (self,) }
		
		result = self.__class__(**kwargs)
		
		def _backward():
			self.grad += result.data * (1 - result.data) * result.grad 
		
		result._backward = _backward
		return result
					
	def __neg__(self):
		return self * -1
	
	def __radd__(self, other):
		return self + other
	
	def __sub__(self, other):
	    return self + (-1*other)
	
	def __rsub__(self, other): 
	    return other + (-self)
	
	def __rmul__(self, other): 
	    return self * other
	
	def __truediv__(self, other): 
	    return self * other**-1
	
	def __rtruediv__(self, other):
	    return other * self**-1
	
	def __repr__(self):
		return f"(data:{self.data}, grad:{self.grad})"	


		
if __name__ == '__main__':
		v= Scalar(5)
		print(v)
		
		