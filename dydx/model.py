# modelling 
import pickle
from layers import Linear, Loss, Layer
from linear_algebra import Array
from pathlib import Path 
import random
from dydx import Scalar
from collections import Counter 
file= 'encoded_records.list'

p = Path(__file__).parent / 'data'

with open( p / file, 'rb') as fh:
	data=  pickle.load(fh) 
	


# ID,Age,Agency,Agency Type,Commision (in value),Destination,Distribution Channel,Duration,Gender,Net Sales,Product Name,Claim

x =['Age','Agency','Agency Type','Commision (in value)','Destination','Distribution Channel','Duration,Gender','Net Sales','Product Name']
y=  ['Claim']
xvals = lambda r: [v for (k,v) in r.items() if k in x]
yvals = lambda r: [v for (k,v) in r.items() if k in y]
data = [  (xvals(r), yvals(r)) for r in data ]


dist = Counter((y[0] for (_,y) in data))
# balancing data
#print(dist[1],dist[0])
pos =  [  (x, y)  for (x,y) in data if y[0] ==1.0 ][:dist[1]]
neg =  [  (x, y) for (x,y) in data if y[0] ==0.0  ][:dist[1]]
data = pos + neg 
random.shuffle(data)
#print(len(data))
# data transformatiom 
minmax= []
dt = []
for i in range(len(data[0][0])):
	m= [p[i] for p in [x for (x,_) in data]]
	if min(m) == 0 :
		dt.append((min(m),max(m)))
	else:
		mu = (1/len(m)) * sum(m)
		var = (1/(len(m)-1) ) * sum( [(mu - x)**2 for x in m])
		dt.append((mu, var**(1/2)))
		
def normalise(x, mu, std):
	return (x-mu)/std
def standardise(x, min, max):
	return (x-min)/(max-min)

data =[  ([ Scalar(standardise(x, *m)) if m[0]== 0 else Scalar(normalise(x, *m)) for (x,m) in zip(xx,dt)], Scalar(float(y[0])))  for (xx,y)  in data ][:160]


# train, val, test split 
train = data[:int(len(data)*0.70)]
val = data[int(len(data)*0.70): int(len(data)*0.90)]
test =data[int(len(data)*0.90):]
print(len(train))
#print(len(val))
#print(len(test))
#print(test)

def flatten(v):
    for i in v:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i
            
def batch(data, n=16):
    l = len(data)
    for ndx in range(0, l, n):
        x,y = [],[]
        for (_x,_y) in data[ndx:min(ndx + n, l)]:
        	x.append(_x)
        	y.append(_y)
        yield Array(values=x),Array(values=y)



l1 = Linear( random=True, dims=(8, 64))
l2 = Linear( random=True,dims=(64, 64))
l3 = Linear( random=True,dims=(64, 16))
l4 = Linear( activation=False, random=True, dims=(16,1)
)

		
names = ["l1","l2","l3","l4"]		
layers =[l1,l2,l3,l4]
ls = dict(zip(names,layers))
		


class Model(Layer):
	
	def __init__(self, layers):
		self.ln = layers.keys()
		
		for key in layers:
			setattr(self, key, layers[key])
	
	def parameters(self):
	 	params = []
	 	for n in self.ln:
	 		params += getattr(self,n).parameters()
	 	return params 
	 	
	 	
	def __call__(self,x ):
#		for layer in self.layers:
#				x = layer(x)
		for n in self.ln:
			x= getattr(self,n)(x)
		return x
		
	def zero_grad(self):
		for n in self.ln:
			getattr(self, n)._zero_grad()
		

model = Model(ls) 
loss = Loss()
EPOCHS= 100

for epoch in range(EPOCHS):
	print("Training".center(70,"#"))
	for (idx,xy) in enumerate( batch(train, 32)):
		x,y = xy 
		model.zero_grad()	
		y_hat =model(x) 
		
		acc, avg_loss = loss(y,y_hat)
		avg_loss.backward()
		
		print(f'Epoch {epoch}:{idx} Loss:{avg_loss.data} train accuracy:{acc*100}%')
			
		lr = 0.0005 
		for p in model.parameters():
				# print('grads',l.grad())
				# print('p.grad', p.grad)				
			p.data = p.data -  lr * p.grad
		
		
	print("Validation".center(70,"#"))
	for (idx,xy) in enumerate( batch(val, 32)):
		x,y = xy 
#			for layer in model.layers:
#				x = layer(x)
		y_hat = model(x)
		acc, avg_loss= loss(y,y_hat)
		print(f'Epoch {epoch}:{idx} Loss:{avg_loss.data} val accuracy:{acc*100}%')	

# train()
