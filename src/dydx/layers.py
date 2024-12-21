from .linear_algebra import Array 
from .autodiff import Scalar 
import random 
from functools import reduce
import sys
import logging 

# gather seed information 
# seed = random.randint(1, sys.maxsize)
# rdm = random.seed(seed)

# seed = 8433248431303025905 # 3786509482071647239 # _
# rdm = random.seed(seed)


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class Layer:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    

class Embedding(Layer):
    def __init__(self, activation=True, dims=None, values=None, seed=None, logging=True):
        super().__init__()
        if seed is None:
            seed = random.randint(1, sys.maxsize)
            
        self.dims = dims
        self.seed = seed
        random.seed(seed)
        self.random = random
        # self.random.seed(seed)
        if logging:
            logger.info(f'Layer seed: {seed}')
        
        if values is None:
            self._init_weights()
        else:
            self.wb= Array(values=values)
        self.seed = seed 
        self.activation = activation       
        
    def _init_weights(self):
        x= (6/sum(self.dims))**(1/2)
        self.wb =  Array( values = [ [Scalar(self.random.uniform(-x,x)) for _ in range(self.dims[1] )] for _ in range(self.dims[0])] )
    
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
    def __init__(self, activation=True, dims=None, values=None, logging=True, seed=random.randint(1, sys.maxsize)):

        super().__init__()
        self.dims = dims
        self.seed = seed 
        random.seed(seed) 
        self.random = random
        if logging:
            logger.info(f"Layer seed: {seed}")
  
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
        x_b =  x.stack(Array(values=[[Scalar(1)]] * bd ), dim=0)
        # [[xs],
        #  [1] ] 
        
        out = x_b @ self.wb
        if self.activation:
            out.values = [[x.relu() for x in row] for row in out.values] 
        else:
            out.values = [[x.sigmoid() for x in row] for row in out.values]
        return out
        
#####
    def parameters(self):
        return [p for row in self.wb.values for p in row] 	
######
    
    def _init_weights(self):
        # xavier init
        x= (6/sum(self.dims))**(1/2)
        self.wb =  Array( values = [ [Scalar(self.random.uniform(-x,x)) for _ in range(self.dims[1] )] for _ in range(self.dims[0] +1)] )
  
  
    def __call__(self, x):
        return self.forward(x)
