from linear_algebra import Array
from dydx import Scalar
from random import randint

# 4x5 matrx, with bias 4x1

def w_b( dims=(4,5)):
	return Array(values=[[Scalar(randint(-10,10)/10) for _ in range(dims[1])] for _ in range(dims[0]) ]), Array(values=[[Scalar(0)] for _ in range(dims[1])])
	

w,b = w_b()
print('weights\n',w)
print('biases\n',b)


x = Array(random=True, dims=(4,1))
print(x)

r=  w.T() @ x + b
