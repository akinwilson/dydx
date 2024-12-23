from .layers import Linear, Layer
from .metrics import Loss
from .dataset import Dataset


class Model(Layer):
	
	def __init__(self, layers):
		self.ln = layers.keys()
		
		for key in layers:
			setattr(self, key, layers[key])
	

	def __call__(self,x):
		for n in self.ln:
			x = getattr(self,n)(x)
		return x
	
 
	def parameters(self):
		params = []
		for n in self.ln:
			x = getattr(self,n).parameters()
			params += x
		return params 


	def seeds(self):
		sds = [getattr(self, n).seed for n in self.ln]
		return sds


	def zero_grad(self):
		for n in self.ln:
			getattr(self, n).zero_grad()


	def step(self):
		for n in self.ln:
			l = getattr(self, n).step()
			setattr(self, n, l)
		return self


     

		


if __name__ == "__main__":
	l1 = Linear( random=True, dims=(8, 64))
	l2 = Linear( random=True,dims=(64, 64))
	l3 = Linear( random=True,dims=(64, 16))
	l4 = Linear( activation=False, random=True, dims=(16,1)
	)

	names = ["l1","l2","l3","l4"]
	layers =[l1,l2,l3,l4]
	ls = dict(zip(names,layers))

	model = Model(ls) 