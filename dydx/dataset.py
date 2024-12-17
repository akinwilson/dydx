import pickle

from .linear_algebra import Array
from pathlib import Path 
import random, sys

# gather seed information 
seed = random.randint(1, sys.maxsize)
rdm = random.seed(seed)

# seed =  5519695473683192700 # 2061982751930337971
# rdm = random.seed(seed)

print(f"dataset seed: {seed}")


from .dydx import Scalar
from collections import Counter, OrderedDict

file= 'encoded_records.list'

p = Path(__file__).parent / 'data'

fp = p / file



xcols =['Age','Agency','Agency Type','Commision (in value)','Destination','Distribution Channel','Duration','Gender','Net Sales','Product Name']
xcols_types = ['ordinal', 'categorical','binary','ordinal', 'categorical','binary', 'ordinal','cateogrical', 'ordinal', 'categorical']
ycols=  ['Claim']

class Dataset:
	
		def __init__(self, filepath=fp, x_names=xcols,y_names=ycols,balance=True,shuffle=True, testing=True):
			'''
			asuming data is loaded as a list of records; 
			[ {'uuid': 1, 'col1':data1,'col2':data2, ... ,'coln':datan},
			  {'uuid': 2, 'col1':data1,'col2':data2, ... ,'coln':datan},
			   .... ]
			'''
			self.filepath= filepath 
			self.x_names = x_names
			self.y_names = y_names
			self.to_balance = balance
			self.shuffle = shuffle
			self.testing = testing
			self.seed = seed 

			def load(filepath):
				with open( filepath, 'rb') as fh:
					return  pickle.load(fh) 
			
			self.data = load(filepath)
			self.records_to_list()

			if self.to_balance:
				self.balance()
			else:
				if self.shuffle: 
					random.shuffle(self.data)
			

			self.scale()
			
			self.train = self.data[:int(len(self.data)*0.70)]
			self.val = self.data[int(len(self.data)*0.70): int(len(self.data)*0.90)]
			self.test = self.data[int(len(self.data)*0.90):]	

		
		def records_to_list(self):
			xvals = lambda r: [v for (k,v) in r.items() if k in self.x_names]
			yvals = lambda r: [v for (k,v) in r.items() if k in self.y_names]
			self.data = [  (xvals(r), yvals(r)) for r in self.data ]

		def balance(self):
			'''
			works for binary targets, finds shortest target group lengths and scales down to smallest group
			'''
			dist = Counter([y[0] for (_,y) in self.data])

			dist = sorted(dist.items(), key=lambda kv: kv[1])
			max_vals_ys = dist[:1]

			pos =  [  (x, y)  for (x,y) in self.data if y[0] ==1.0 ][:max_vals_ys[0][1]]
			neg =  [  (x, y) for (x,y) in self.data if y[0] ==0.0  ][:max_vals_ys[0][1]]

			self.data =  pos + neg 
			random.shuffle(self.data)
			
		def __getitem__(self,idx, split=['train','val','test'][1]):
			return getattr(self,split)[idx]
		
		def __len__(self,split=['train','val','test'][1]):
			return len(getattr(self,split))
		
		def scale(self):
			dt = [] # data tranformation 
			for i in range(len(self.data[0][0])):
				# collect ith predictor
				m= [p[i] for p in [x for (x,_) in self.data]]
				if min(m) == 0 and xcols_types[i] == 'ordinal': # asumming standardisation min max scaling
					dt.append((min(m),max(m)))

				elif min(m) == 0 and (xcols_types[i] == 'categorical' or 'binary'):
					dt.append(('cateogrical'))
				else: # assuming normalsation 
					mu = (1/len(m)) * sum(m)
					var = (1/(len(m)-1) ) * sum( [(mu - x)**2 for x in m])
					dt.append((mu, var**(1/2)))
					
			def normalise(x, mu, std):
				return (x-mu)/std
			def standardise(x, min, max):
				return (x-min)/(max-min)

			data = []
			for (xx,y) in self.data:
				row = []
				for (idx,x) in enumerate(xx):

					if len(dt[idx]) == 2:
						if dt[idx][0]== 0:
							row.append( Scalar(standardise(x, *dt[idx])))
						else:
							row.append(Scalar(normalise(x, *dt[idx])))
					else:
						row.append(Scalar(x))
				data.append((row, Scalar(float(y[0])) ))
					# data.append(([ Scalar(standardise(x, *m)) if m[0]== 0 else Scalar(normalise(x, *m)) for (x,m) in zip(xx,dt)], Scalar(float(y[0]))))
    
			# self.data = [  ([ Scalar(x) if m[0]== 0 else Scalar(normalise(x, *m)) for (x,m) in zip(xx,dt)], Scalar(float(y[0])))  for (xx,y)  in self.data ]
			self.data = data 
   			# only use 160 examples for testing the pipeline
			self.data = self.data[:512] if self.testing else self.data 		

		
class DataLoader:
	def __init__(self, dataset, batch_size):
		self.data = dataset
		self.batch_size = batch_size

	def __call__(self):
		l = len(self.data)
		for ndx in range(0, l, self.batch_size):
			x,y = [],[]
			for (_x,_y) in self.data[ndx:min(ndx + self.batch_size, l)]:
				x.append(_x)
				y.append(_y)
			yield Array(values=x),Array(values=y)



if __name__ == "__main__":	
	ds = Dataset()
	splits = ['train','val','test']
	print(f"dataset sizes for {', '.join(splits)} are {[ds.__len__(split) for split in splits]} respectively.")
	dst = ds.train
	dsv = ds.val
	dss = ds.test
	# iterators 
	dltrain  = DataLoader(dst,16)
	dlval = DataLoader(dsv,16)
	dltest = DataLoader(dss,16)



	for (idx,xy) in enumerate(dltrain()):
		x,y = xy
		print(f'{idx}')
# ID,Age,Agency,Agency Type,Commision (in value),Destination,Distribution Channel,Duration,Gender,Net Sales,Product Name,Claim

# x =['Age','Agency','Agency Type','Commision (in value)','Destination','Distribution Channel','Duration,Gender','Net Sales','Product Name']
# y=  ['Claim']
# xvals = lambda r: [v for (k,v) in r.items() if k in x]
# yvals = lambda r: [v for (k,v) in r.items() if k in y]
# data = [  (xvals(r), yvals(r)) for r in data ]


# dist = Counter((y[0] for (_,y) in data))

# dist = sorted(dist.items(), key=lambda kv: kv[1])
# max_vals_ys = dist[:1]

# pos =  [  (x, y)  for (x,y) in data if y[0] ==1.0 ][:max_vals_ys[0][1]]
# neg =  [  (x, y) for (x,y) in data if y[0] ==0.0  ][:max_vals_ys[0][1]]


# data = pos + neg 
# random.shuffle(data)
# #print(len(data))
# # data transformatiom 
# minmax= []
# dt = []
# for i in range(len(data[0][0])):
# 	m= [p[i] for p in [x for (x,_) in data]]
# 	if min(m) == 0 :
# 		dt.append((min(m),max(m)))
# 	else:
# 		mu = (1/len(m)) * sum(m)
# 		var = (1/(len(m)-1) ) * sum( [(mu - x)**2 for x in m])
# 		dt.append((mu, var**(1/2)))
		
# def normalise(x, mu, std):
# 	return (x-mu)/std
# def standardise(x, min, max):
# 	return (x-min)/(max-min)

# data =[  ([ Scalar(standardise(x, *m)) if m[0]== 0 else Scalar(normalise(x, *m)) for (x,m) in zip(xx,dt)], Scalar(float(y[0])))  for (xx,y)  in data ][:160]


# # train, val, test split 
# train = data[:int(len(data)*0.70)]
# val = data[int(len(data)*0.70): int(len(data)*0.90)]
# test =data[int(len(data)*0.90):]
# print(len(train))
# #print(len(val))
# #print(len(test))
# #print(test)

# def batch(data, n=16):
#     l = len(data)
#     for ndx in range(0, l, n):
#         x,y = [],[]
#         for (_x,_y) in data[ndx:min(ndx + n, l)]:
#         	x.append(_x)
#         	y.append(_y)
#         yield Array(values=x),Array(values=y)
