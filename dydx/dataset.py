import pickle

from linear_algebra import Array
from pathlib import Path 
import random
from dydx import Scalar
from collections import Counter, OrderedDict
file= 'encoded_records.list'

p = Path(__file__).parent / 'data'

with open( p / file, 'rb') as fh:
	data=  pickle.load(fh) 

class Dataset:
	
		def __init__(self, path, x_names,y_names,balance=True,shuffle=True):
			pass
		
		def load(self):
			pass
			
		def __getitem__(self,idx):
			pass
		
		def __len__(self):
			pass
		
		def _scale(self):
			pass

class DataLoader:
	def batch(self,batch_size):
		pass
		
# ID,Age,Agency,Agency Type,Commision (in value),Destination,Distribution Channel,Duration,Gender,Net Sales,Product Name,Claim

x =['Age','Agency','Agency Type','Commision (in value)','Destination','Distribution Channel','Duration,Gender','Net Sales','Product Name']
y=  ['Claim']
xvals = lambda r: [v for (k,v) in r.items() if k in x]
yvals = lambda r: [v for (k,v) in r.items() if k in y]
data = [  (xvals(r), yvals(r)) for r in data ]


dist = Counter((y[0] for (_,y) in data))
# balancing data
#print(dist[1],dist[0])



dist = sorted(dist.items(), key=lambda kv: kv[1])
max_vals_ys = dist[:1]

pos =  [  (x, y)  for (x,y) in data if y[0] ==1.0 ][:max_vals_ys[0][1]]
neg =  [  (x, y) for (x,y) in data if y[0] ==0.0  ][:max_vals_ys[0][1]]


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

def batch(data, n=16):
    l = len(data)
    for ndx in range(0, l, n):
        x,y = [],[]
        for (_x,_y) in data[ndx:min(ndx + n, l)]:
        	x.append(_x)
        	y.append(_y)
        yield Array(values=x),Array(values=y)
