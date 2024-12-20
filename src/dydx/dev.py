import random
from functools import reduce
# dims (4,2,3)  -> 
x = [[[1,1,6],[1,1,6]],[[1,1,6],[1,1,6]], [[1,1,6],[1,1,6]],[[1,1,6],[1,1,6]] ]
#  dims (1,3,2) -> 
y = [[[1,1],[1,1],[1,1]]]
#  dims (2,1,3) ->
z =  [[[1,1,1]],[[1,1,1]]]
# dims (2,3,1)  -> 
w = [ [[1],[1],[1]],[[1],[1],[1]]]
#  dims (3,1,2) ->  
u= [[[1,1]],[[1,1]],[[1,1]]]
#  dims (3,2,1) ->  
v = [[[1],[1]],[[1],[1]],[[1],[1]]]



def extract_dims(v,dims=[]):
	try:
		for d in v[:1]:
			dims.append(len(v))
			dims= extract_dims(d,dims)
	except TypeError:
		pass 
	return dims
	
	
dims = []		
dims = extract_dims(x, dims) 					
# print(dims )

def _rand_values(dims):
    p = reduce(lambda x,y: x*y,dims)
    v = [random.randint(0,10) for _ in range(p)]
    
    def reshape(v, dims):
        if len(dims) == 1:
        	return v
        n = reduce(lambda x,y: x*y, dims[1:])
        return [reshape(v[i*n:(i+1)*n], dims[1:]) for i in range(len(v)//n)]

    return reshape(v,dims)

def reshape(v, dims):
	if len(dims) == 1:
	     return v
	n = reduce(lambda x,y: x*y, dims[1:])
	return [reshape(v[i*n:(i+1)*n], dims[1:]) for i in range(len(v)//n)]
	



dims = (32,16,3)
print('supplied dims', dims)
val = _rand_values(dims) 



def permute(val, new_dim):
	
	for 
	pass 
	
val = permute(val, 0,2)



dims = extract_dims(val)
print('returned dims', dims )



def flatten(v):
    for i in v:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i			
			
val = _rand_values(dims)

dims = dims 
print('vals', val) 
fl = list(flatten(val ))
print('flattened',fl)


dims = extract_dims(val,dims)
#print(dims)
from pprint import pprint
t = [[1,2,3],[4,5,6]]
pprint(t)
pprint(reshape(list(flatten(t )), (3,2)))

