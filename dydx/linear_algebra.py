from random import randint
from typing import Tuple, List
from math import isclose
from functools import reduce
# matrix
#  [ [1,1] ] -> dims (1,2)
# [ [1], [1] ]  -> dims (2,1)

# tensor notation
# dims (1,2,3)  -> [[[1,1,1],[1,1,1]]]
#  dims (1,3,2) -> [[[1,1],[1,1],[1,1]]]
#  dims (2,1,3) -> [[[1,1,1]],[[1,1,1]]]
# dims (2,3,1)  -> [ [[1],[1],[1]],[[1],[1],[1]]]
#  dims (3,1,2) ->  [[[1,1]],[[1,1]],[[1,1]]]
#  dims (3,2,1) ->  [[[1],[1]],[[1],[1]],[[1],[1]]]


class Array:
    
    def __init__(self, values : List[List]=None, random : bool =False, dims : Tuple[int,int]=None):
        
        
        if values is not None and random:
            raise ValueError(f'array {values} provided but randomised array set to {random}')
            
            
        if random:
            if dims is None:
                raise ValueError('Require dimensions of random array to be instantiated')
            if any( x < 1 for x in dims ):
                raise ValueError(f'Require dimensions of array to be positive, got {dims}')
            self.dims = dims
            
            def _rand_values(dims):
            	p = reduce(lambda x,y: x*y,dims)
                v = [ randint(0,10) for _ in range(p)]
            	
            	def reshape(v, dims):
            		if len(dims) == 1:
            			return v
            		n = reduce(lambda x,y: x*y, dims[1:])
            		return [reshape(v[i*n:(i+1)*n], dims[1:]) for i in range(len(v)//n)]
            		
            	return reshape(v,dims)
            self.values = _rand_values(dims)
            
            
        if values is not None:
            dims = []
            def extract_dims(v, dims):
            	try:
            		for d in v[:1]:
            			dims.append(len(v))
            			dims= extract_dims(d,dims)
            	except TypeError:
            		pass
            	return dims
            dims = extract_dims(values, dims)
            self.dims =dims
            self.values = values
 
 
 

    def extract_dims(self, v, dims=[]):
            try:
            	for d in v[:1]:
            		dims.append(len(v))
            		dims=self.extract_dims(d,dims)
            except TypeError:
            	pass
            return dims       
	 
    def __repr__(self):
        s = ''
        for x in self.values:
            s += x.__repr__() + ',\n'
        s = '['+ s.rstrip(',\n') + ']'
        return s
   
        
    def __matmul__(self,other):
#        if type(self) != type(other):
#            raise TypeError(f'Require two objects of type array to multiply together. Got {self} and {other}')
        if self.dims[-1] != other.dims[0]:
             raise ValueError(f'Dimension of array do not match. Got {self.dims} and {other.dims}')
             
        result = self.__class__(random=True, dims=(self.dims[0],other.dims[-1]))
        
        for xi in range(result.dims[0]):
            for yi in range(result.dims[-1]):
                r = self.values[xi]
                c = [other.values[i][yi] for i in range(len(r))]
                result.values[xi][yi] = sum(x*y for (x,y) in zip(r,c))
        return result
   
        
                  
    def __eq__(self, other):
        if type(self) != type(other):
                TypeError(f'Cannot compare different types. Got {type(self)} and {type(other)}')
        if self.dims != other.dims:
            return False
        ijs = [ (i,j) for i in range(self.dims[0]) for j in range(self.dims[1])]
        return all([ isclose(self.values[i][j], other.values[i][j], abs_tol=1e-6) for (i,j) in ijs ])
        
        
    def __ne__(self, other): 
        if type(self) != type(other):
                TypeError(f'Cannot compare different types. Got {type(self)} and {type(other)}')
        if self.dims != other.dims:
            return True         
        ijs = [ (i,j) for i in range(self.dims[0]) for j in range(self.dims[1])]
        return any([ not isclose(self.values[i][j], other.values[i][j], abs_tol=1e-6) for (i,j) in ijs ])
        
                                  
    def _square(self):
        return all([i == self.dims[0] for i in self.dims])
    
    
    def _vector(self):
        return True if 1 in self.dims and len(self.dims) == 2 else False

    def _matrix(self):
        return True if  len(self.dims) == 2 and all( x>=2 for x in self.dims) else False
                        
        
    def __add__(self, other):
        
#        if type(self) != type(other):
#            TypeError(f'Cannot add different types. Got {type(self)} and {type(other)}')          
        result = self.__class__(random =True,dims=self.dims)._zeros()
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                result.values[i][j] += self.values[i][j] + other.values[i][j] 
        return result
        
        
    def __sub__(self, other):
#        if type(self) != type(other):
#            TypeError(f'Cannot subtract different types. Got {type(self)} and {type(other)}')
        result = self.__class__(random =True,dims=self.dims)._zeros()
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                  result.values[i][j] += self.values[i][j] - other.values[i][j] 
        return result
        
        
    def __rsub__(self, other):
#        if type(self) != type(other):
#            TypeError(f'Cannot subtract different types. Got {type(self)} and {type(other)}')
        result = self.__class__(random =True,dims=self.dims)._zeros()
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                  result.values[i][j] += other.values[i][j] - self.values[i][j]
        return result        
        
                        
                       
    def __pow__(self,exponent):
         if not self._square():
            raise ValueError(f'Cannot exponentiate non-square matrix')
         if type(exponent) != type(int):
             NotImplementedError(f'Need to implement non-integer exponentiation')
         # dealing with integers first

         if exponent == 0:
             return self.identity()
         if exponent == 1:
             return self
              
         if exponent == -1:
             return self.inverse()
             
         if exponent >1:
             result = self
             for _ in range(exponent - 1):
                 result =  result @ self
             return result
             
         if exponent < -1:
             m_inv = self.inverse()
             for _ in range( abs(exponent) -1):
                   m_inv = m_inv @ self.inverse()
             return m_inv
    
                            
    def normalise(self, dim: int = None, norm : int  = None):
        # normilsation is vector-wise normalisation and not matrix-wise 
        if dim is None:
            ValueError('To normalise, the order (rows equals 0, columns equal 1, depth 2 etc) to respect must be given')
        if norm is None:
            ValueError('To normalise, the norm type must be given. I.e zeroth norm (0) absolute norm (1), euclidean norm (2), etc')
        result = []
        if norm == 0:
          if dim == 0:
              m = self.values
          if dim == 1:
              m = self.T().values
          for row in m:
              result.append([ 1 if i == row.index(max(row)) else 0 for i in range(len(row ))])
          return self.__class__(values=result)
        
        n_factor = lambda vec,norm : (sum([x**(norm) for x in vec])**(1/norm))**(-1)
        if norm >= 1:
            if dim == 0:
                m = self.values 
                for row in m:
                    result.append([n_factor(row,norm) * val for val in row])
                return self.__class__(values=result)
                
            if dim == 1:
                m = self.T().values
                for row in m:
                    result.append([n_factor(row,norm) * val for val in row])
            return self.__class__(values=result).T()
            
            
           
           
    # def __iter__(self):
    #     self.ij = (0,0)
    #     return self

    # def __next__(self): 
    #     if self.ij[0] > self.dims[0] or self.ij[1] > self.dims[1]:
    #         raise StopIteration
    #     else:
    #         v = self.values[self.ij[0]][self.ij[1]]
    #         self.ij[0] += 1 
            
            
              
          
    def T(self):
        result = []
        for xi in range(self.dims[1]):
            row = []
            for yi in range(self.dims[0]):
                row.append(self.values[yi][xi])
            result.append(row)    
        return self.__class__(values=result)

                
    def identity(self):
        # square case only
        if not self._square():
            raise NotImplementedError(f'Have not implemented identity equivalent for none-square matrix') 
        I = [[0] * self.dims[1] for _ in range(self.dims[0])]
        j = 0
        for i in range(len(I)):
            I[i][j] = 1
            j+=1
        return self.__class__(values=I)
    
    
    def diagonal(self):
        # return purely diagonal
        if not self._square():
            raise NotImplementedError(f'Diagonalisation of array for non square array not implemented yet') 

    def __mul__(self, value):
        m = self.values
        m = [[ij*value for  ij in row] for row in m ]
        return self.__class__(values=m) 

    def __rmul__(self, value):
        m = self.values
        m = [[ij*value for  ij in row] for row in m ]
        return self.__class__(values=m) 
            
    def _zeros(self):
        idxs =  [ (x,y) for x in range(self.dims[0]) for y in range(self.dims[1])] 

                
        for (xi,yi) in idxs:
            self.values[xi][yi] = 0
        return self.__class__(values=self.values)


    def determinant(self):
        # can inverse be calculate? det
        if not self._square():
            raise ValueError(f'Cannot calculate determinant on non-square array. Got dimensions {self.dims}')
            
        m = self.values
        # base case
        if len(m) == 2: # (2 x 2 ) array 
            D = m[0][0] * m[1][1]  - m[0][1] * m[1][0]
            return D       
        D = 0

        for c in range(len(m)):
            D += ((-1)**c)*m[0][c]*self._minor(0,c).determinant() 
       
        return D
            
    def stack(self,other,dim:int=0):
        
#        if type(self) != type(other):
#            raise TypeError(f'Cannot stack arrays of two objects of different types. Got {type(self)} and {type(other)}')
        if dim not in set(range(len(self.dims))):
             raise ValueError(f'Given dim to stack over is out it range. got dim number {dim} but possible dims are {set(range(len(self.dims)))}')
        if self.dims[dim] != other.dims[dim]:
             raise ValueError(f'The dimensions to stack over do not match. got dims {self.dims[dim]} and {other.dims[dim]}')
        if dim == 0:
             v1,v2 = self.values, other.values
             result = [ [*v,*u] for (v,u) in zip(v1,v2)]
             return self.__class__(values=result)
        if dim ==1:
             v1,v2 = self.T().values, other.T().values
             result = [ [*v,*u] for (v,u) in zip(v1,v2)]
             return self.__class__(values=result).T()
   
    def _minor(self, i, j):
        m = self.values
        # removing row
        m = m[:i] + m[i + 1:] 
        # removing col
        m = [x[:j] + x[j + 1:] for x in m]
        return self.__class__(values=m) 
                   
    def inverse(self):
        D= self.determinant()
        if D == 0:
            raise ZeroDivisionError(f'Determinant for array is equal to {0} therefore cannot find inverse.')
        m = self.values
        #case for 2x2 matrix:
        if len(m) == 2:
            values =  [[m[1][1]/D -1*m[0][1]/D],
                [-1*m[1][0]/D, m[0][0]/D]]
            return self.__class__(values=values)

        #find matrix of cofactors
        c = []
        for i in range(len(m)):
            c_row = []
            for j in range(len(m)):
                minor = self._minor(i,j)
                c_row.append(((-1)**(i+j)) * minor.determinant())
            c.append(c_row)
        
        c = self.__class__(values=c).T()
    
        for i in range(c.dims[0]):
            for j in range(c.dims[1]):
                c.values[i][j] = c.values[i][j]/D
        return c

                
    def eigenvectors(self, k:int=20):
        # power iteraction method
        e_vals= self.eigenvalues()
        e1 = [[1]] + [[0]]*(self.dims[1]-1)
        print(e1)
        e_vec1 = self.__class__(values=e1)
        e_vecn = self.__class__(values=e1)
        e_vecs = []
        #eigen vector of largest e-value
        for _ in range(k):
            e_vec1 = self @ e_vec1
            e_vec1 = e_vec1.normalise(dim=1,norm=2)
        print('Vector for largest eigen value')
        print(e_vec1)
        print()
        # eigen vector of smallest e-value
        self_inv = self.inverse()
        for _ in range(k):
            e_vecn = self_inv @ e_vecn
            e_vecn = e_vecn.normalise(dim=1,norm=2)
        print('Vector for smallest eigen value')
        print(e_vecn)
        print()
           
            
       
        
    
   
    def eigenvalues(self):
        # the self = QR 
        # gram-schmidt process for
        # constructing orthonormal basis 
        # treating matrix as column vectors
        ncols, nrows = self.dims
        
        # need to use dims on init to check whether transpose of values is required 
        # make sure dims are updated when 
        # transpose operation takes places 
        A = [self.__class__(values=[row], dims=(nrows,1)).T() for row in self.T().values]

 
        proj_a_on_u  = lambda u,a : (u.T() @ a ).values[0][0] / (u.T() @ u).values[0][0] * u
         
        norm_u = lambda u : u.normalise(dim=1,norm=2)
              
        us = []
        
        def kth_u(a,u,k):
            # base case
            if k == 0:
                u = a[k]
                return u
            else:
                result = self.__class__(random=True,dims=(nrows,1))._zeros()
                t = [proj_a_on_u(v,a[k]) for v in u]
                for v in t:
                    result += v
                u = a[k] - result
                return u
       
        for k in range(ncols):
            us.append(kth_u(A,us,k))

        es = [norm_u(u) for u in us]         
        Q = es[0]
        for v in es[1:]:
            Q = Q.stack(v,dim=0)                    
        offset = []
        R = []
        
        for (i,e) in enumerate(es):
            rrow =[]
            for a in A[i:]:
                rrow.append((a.T() @ e ).values[0][0])
            rrow = offset + rrow
            R.append(rrow)
            offset = [0] + offset 
                    
        R = self.__class__(values = R)
        eigenvalues = [R.values[i][i] for i in range(ncols)]
        print('R', R)
        print('Q', Q)
        print('self', self)
        print('Q@R',Q@R )
        # failing at the moment but not top sure why
        assert self == Q@R , 'reconstruction of matrix using Q, an orthonormal matrix, and R, an upper triangular matrix, has failed'
        # Q is an orthonormal basis 
        # R is an upper triangular matrix
        #print('R\n', R)
        #print('Q@R\n', Q @ R)
        #print('Q.T @ A\n',Q.T() @ self)
        return sorted([e/max(eigenvalues) for e in eigenvalues], reverse=True)
   
             
    def spectral_decomposition(self):
        if self._vector():
            raise ValueError(f'decomposition algorithms like spectral decomposition apply to matrices, got vector of dimensions {self.dims} instead')
            
        if not self._square():
            raise ValueError(f'To decompose non singular (i.e full rank) array using spectral decomposition, we require a square array, this array has dimensions {self.dims}. Try to apply singular value decomposition instead for non-square arrays')
        pass
        
        
    def singular_value_decomposition(self):
        if self._vector():
            raise ValueError(f'decomposition algorithms like singular value decompositions apply to matrices, got vector of dimensions {self.dims} instead')
                    
        if self._square():
            raise ValueError(f'To decompose non singular (i.e full rank) array using singular value decomposition, we require a rectangular array, this array has dimensions {self.dims}. Try to apply spectral decomposition instead for square arrays')
        pass         
        
     
        # aggregate = lambda x,y : x.stack(y,dim=0)
        # reduce(aggregate(norm_basis[0], norm_basis[1:]))
        # print(proj_u)
        
        # print('u', u)
        # unorm = u.normalise(dim=1,norm=2)
        # print('u.norm' , unorm)
        
       
        
        
        
        
        
    
        
    
                
                
if __name__ == '__main__':
	
	m1 = Array(random=True, dims=(4,2))
	m2 = Array(random=True, dims=(2,3))
	s = Array(values=[[3,0,0],[0,5,0],[0,0,5]], dims=(4,4))
	I= Array(values=[[1,0,0],[0,1,0],[0,0,1]], random=False)
	print('m1\n',m1)
	print('m1.T\n', m1.T())
	print('m2\n',m2)
	print( m1 == m1)
	print(m1 != m1)
	print( 'm1 @ m2\n', m1 @ m2)
	print( 'm1 @ m2 @ I \n', m1 @ m2 @ I)
	x = 3- 2j
	print(m1._square())
	print(m1._zeros(), m1.dims)
	print('s\n', s)
	print('s._minor(0,0)\n', s._minor(0,0))
	print('s._minor(0,2)\n', s._minor(0,2))
	print('s._minor(1,2)\n', s._minor(1,2))
	print('s.determinant()\n', s.determinant())
	print(s.inverse() @ s)
	m3 = Array(random=True, dims=(5,5))
	m3m3inv = m3.inverse() @ m3
	print('m3.inverse() @ m3\n' , m3m3inv)
	print(m3m3inv == m3.identity() )
	m4 = Array(random=True, dims=(3,3)).identity()
	print(f'3 * {m4}=\n', 3 * m4)
	m5 = Array(random=True, dims=(3,3))
	print(f'm5**2 {m5**2}')
	assert m5**3 == m5 @ m5 @ m5
	assert m5**-1 == m5.inverse()
	assert m5**(-3) == m5**-1 @ m5**-1 @ m5**-1
	print('vector wise normalisation')
	print(f'm5 {m5}')
	m5norm = m5.normalise(dim=0,norm=0)
	print(f'm5.normalise()) {m5norm}')
	print(f'm5norm @ m5norm.T()) {m5norm @ m5norm.T()}')
	m6 = Array(values= [[12,-51,4],[6,167,-68],[-4,24,-41]])
	print('m6\n', m6)
	print('Eigen values')
	# print(m6.eigenvalues())
	#print('eigen vectors')
	#print(m6.eigenvectors())
	#print('stacking')
#print(m6.stack(m6, dim=00))
# print(m5.normalise().T() @ m5.normalise())

#print(dir(m1))