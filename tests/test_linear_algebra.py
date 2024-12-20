import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from linear_algebra import Array 
import pytest 


def test_random_matrix_instantiation():
    m1 = Array(random=True, dims=(4,2))
    m2 = Array(random=True, dims=(2,3))
    assert m1.dims == (4,2)
    assert m2.dims == (2,3)
    
def test_value_passed_matrix_with_dims_provided_instantiation():
    v = [[3,0,0],[0,5,0],[0,0,5]]
    s = Array(values=v, dims=(3,3))
    assert s.values == v
    assert s.dims == (3,3)

def test_negative_dims_with_random_matrix():
    with pytest.raises(ValueError):
       Array(random=True, dims=(-4,4))
def test_value_passed_matrix_with_wrong_dims_provided_instantiation():
    v = [[3,0,0],[0,5,0],[0,0,5]]
    s = Array(values=v, dims=(4,4))
    # need to notified that dims will
    # be assumed from values provided
    assert s.values == v
    assert s.dims == (3,3)
    
def test_3x3_matrix_identity_without_specifying_dims_instantiation():
    v = [[1,0,0],[0,1,0],[0,0,1]]
    I= Array(values=v, random=False)
    assert I.values == v
    assert I.dims == (3,3)

def test_equality_and_not_equality():
    m1 = Array(random=True, dims=(4,2))
    m2 = Array(random=True, dims=(2,3))
    assert m1 == m1 
    assert m1 != m2
    
def test_matrix_by_matrix_multiplication():
    v = [[3,0,0],
           [0,5,0],
           [0,0,5]]
    s1 = Array(values=v)
    v =  [[3,0],
            [0,5],
            [0,0]]
    s2= Array(values=v)
    result = [[9,0 ],
                    [0, 25],
                    [0,0]]
    
    assert (s1@s2).values == result

    

def test_minor_matrices_creation():
    v = [[3,0,0,4],
           [0,5,0,1],
           [0,0,5,8],
           [7,9,4,2]]
    s = Array(values=v, dims=(4,4))
    s00 =  s._minor(0,0)
    s02 = s._minor(0,2)
    s12 =  s._minor(1,2)
    s32 = s._minor(3,2)
    v00 = [
           [5,0,1],
           [0,5,8],
           [9,4,2]]
    v02 =[
           [0,5,1],
           [0,0,8],
           [7,9,2]]
    v12 = [[3,0,4],
           [0,0,8],
           [7,9,2]]
    v32 =[[3,0,4],
           [0,5,1],
           [0,0,8]]
     
    assert v00 == s00.values
    assert v02 == s02.values
    assert v12 == s12.values
    assert v32 == s32.values
       
def test_determinant_creation():
    v = [[1,1,1,1],
           [1,-1,1,0],
           [1,1,0,0],
           [1,0,0,0]]
    s = Array(values=v, dims=(4,4))
    assert s.determinant()  == 1
    
def test_inverse_creation():
    v = [[1,1,1,1],
           [1,-1,1,0],
           [1,1,0,0],
           [1,0,0,0]]
    s = Array(values=v, dims=(4,4))
    assert s.inverse() @  s == s.identity()
    
def test_slicing_array():
    pass

def test_element_extraction():
    pass
 
#   cd code/linear_algebra/tests  && pytest

#print('m1\n',m1)
#print('m1.T\n', m1.T())
#print('m2\n',m2)
#print( m1 == m1)
#print(m1 != m1)
#print( 'm1 @ m2\n', m1 @ m2)

#print( 'm1 @ m2 @ I \n', m1 @ m2 @ I)

x = 3- 2j
#print(m1._square())
#print(m1._zeros(), m1.dims)
s = Array(values=[[1,2,3,4],[3,0,0,4],[0,5,0,1],[0,0,5,8]], dims=(4,4))
print('s\n', s)
print('s._minor(0,0)\n', s._minor(0,0))
print('s._minor(0,2)\n', s._minor(0,2))
print('s._minor(1,2)\n', s._minor(1,2))
print('s.determinant()\n', s.determinant())
print(s.inverse() @ s)
m3 = Array(random=True, dims=(5,5))
m3m3inv = m3.inverse() @ m3
print('m3.inverse() @ m3\n' , m3m3inv  )
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
print(m6.eigenvalues())
print('eigen vectors')
print(m6.eigenvectors())
print('stacking')
#print(m6.stack(m6, dim=00))
# print(m5.normalise().T() @ m5.normalise())

