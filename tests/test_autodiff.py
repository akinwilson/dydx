from dydx.autodiff import Scalar 
import pytest 



def test_forward():
    a= 4 * Scalar(2)
    b = 6*(a**2)
    c = (a * b**4).relu()
    d = ((c-a)**2).sigmoid()
    assert d.data == 2.8634668926570093e-89, f"Incorrect forward propagation. Expected {2.8634668926570093e-89} got {d.data}" 

    
    
def test_backward():
    a= 4 * Scalar(2)
    b = 6*(a**2)
    c = (a * b**4).relu()
    d = ((c-a)**2).sigmoid()
    d.backward()
    assert c.grad == 9.96178229182573e-78, f"Incorrect gradient calculated, Expected {9.96178229182573e-78} received {c.grad}"

    
    
@pytest.mark.skip(reason='gelu function not implemented yet ')
def test_gelu_activation():
    a = Scalar(5)
    a.gelu()
    
