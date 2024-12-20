from .layers import Linear, Layer
from .metrics import Loss
from .dataset import Dataset, DataLoader


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


	def seed(self):
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
	loss = Loss()
	EPOCHS= 100

	ds = Dataset()
	splits = ['train','val','test']
	print(f"dataset sizes for {', '.join(splits)} are {[ds.__len__(split) for split in splits]} respectively.")
	dst = ds.train
	dsv = ds.val
	dss = ds.test
	# iterators 
	dltrain  = DataLoader(dst,16)()
	dlval = DataLoader(dsv,16)()
	dltest = DataLoader(dss,16)()

# for epoch in range(EPOCHS):
# 	print("Training".center(70,"#"))
# 	for (idx,xy) in enumerate( dltrain):
# 		x,y = xy 
# 		model.zero_grad()	
# 		y_hat =model(x) 
		
# 		acc, avg_loss = loss(y,y_hat)
# 		avg_loss.backward()
		
# 		print(f'Epoch {epoch}:{idx} Loss:{avg_loss.data} train accuracy:{acc*100}%')
# 		lr = 0.0005 
# 		for p in model.parameters():
# 			print('p.grad', p.grad)
# 			p.data = p.data -  lr * p.grad
		
		
# 	print("Validation".center(70,"#"))
# 	for (idx,xy) in enumerate( dlval):
# 		x,y = xy 
# #			for layer in model.layers:
# #				x = layer(x)
# 		y_hat = model(x)
# 		acc, avg_loss= loss(y,y_hat)
# 		print(f'Epoch {epoch}:{idx} Loss:{avg_loss.data} val accuracy:{acc*100}%')	

# train()
