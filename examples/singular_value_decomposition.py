# STEPS = 100
# STEP_SIZE = 0.00001
from dydx.linear_algebra import Array
from dydx.utils import flatten
from argparse import ArgumentParser


import math             

mv = [[1,0,0,0,2],
      [0,0,3,0,0], 
      [0,0,0,0,0], 
      [0,2,0,0,0]]

M = Array(values=mv)


def main(args):

    # Will run indefinitely, issues with convergence 
    while True:
        U, S,V = None, None, None 
        try:  
            U,S,V = singular_value_decomposition(M, patience= args.patience, step_size=args.step_size, steps=args.steps)
        except (OverflowError, RuntimeError):
            continue
        finally:
            if U is not None:
                break 

    pass


def singular_value_decomposition(m, patience, step_size, steps):
    '''
    m = U @ S @ V.T()
    
    decompose matrix m of dimensions (m,n) into:
    U: Unitary matrix of (m,m) i.e. U_inv = U.T() and therefore Det(U) = 1 and therefore cols of U form orthonormal basis of m 
    S: Rectangular diagonal matrix of (m,n)
    V: Unitary matrix of (n,n) i.e. V_inv = V.T() and therefore Det(U) = 1 and therefore cols of V form orthonormal basis of n
    
    we use gradient descent in order to approximate U, S and V
    '''
    
    # init random approx for U, S, V 
    U = Array(random=True, dims=(m.dims[0],m.dims[0]), optimisable=True)
    
    S = Array(random=True, dims=m.dims, optimisable=True).diagonal()
    V = Array(random=True, dims=(m.dims[1],m.dims[1]), optimisable=True )
    
    
    # include in loss: loss for non-orthonormality, loss for non-zero off-diagonal elements and finally, for reconstruction approximation, 
    
    def loss_non_orthormality(X):
        sim = X.T() @ X
        # cols span orthonormal basis: sim ----> off-diagonal elements should be zero
        #                  ''              ----> diagonal elements should all be 1 
        sim_diag = sim.diagonal()
        idt = sim.identity()
        non_normal = sim_diag - idt 
        off_diag = sim - sim_diag
        
        l1 = (1/len(list(flatten(off_diag.values)))) *sum([x for x in list(flatten(off_diag.values))])
        l2 = (1/len(list(flatten(non_normal.values)))) *sum([x for x in list(flatten(non_normal.values))])
        return l1 + l2  
    
    def loss_off_diagonal(S):
        S_diag = S.diagonal()
        non_zero = S - S_diag
        return (1/len(list(flatten(non_zero.values)))) * sum([x**2 for x in list(flatten(non_zero.values))])
        
    
    def loss_reconstruction(m, U,S,V):
        return (1/len(list(flatten(( m - (U @ S @ V.T()) ).values)))) * sum([x**2 for x in list(flatten(( m - (U @ S @ V.T()) ).values))])
        
    def func_weighting(f1,f2,f3,f4,w1=1,w2=0.0,w3=0.0,w4=0.0):
        return w1*f1 + w2*f2 + w3*f3 + w4*f4
    
    # Stopping conditions 
    stop = lambda err, patience: True if all([e == err[-patience:][0] for e in err[-patience:]]) else False 
    patience = 5
    err = []
    
    
    for s in range(steps):
        loss = func_weighting(loss_reconstruction(m, U,S,V),
                              loss_off_diagonal(S),
                              loss_non_orthormality(U),
                              loss_non_orthormality(V))
        loss.backward()
        print(f"Step {s} loss: {loss.data}")
        U = U - step_size * U.grad()
        S = S - step_size * S.grad()
        V = V - step_size * V.grad()
        U = U.zero_grad()
        S = S.zero_grad()
        V = V.zero_grad()
        err.append(loss.data)
        if math.isnan(loss.data):
            print("fitting Diverged")
            raise RuntimeError("Loss became infinity and thereby NotANumber; Nan")
        if stop(err, patience) and s > patience + 1: 
            return (U, S, V)


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('-ss', '--step-size', help="step size (equivalent to learnining rate) for optimisation process.", default=0.000001)
    parser.add_argument('-s', '--steps', help="maximum number of steps to take for optimisation processes", default=100)
    parser.add_argument('-p', '--patience', help="steps to wait for before cutting fitting routine", default=5)
    
    args = parser.parse_args()
    
    raise RuntimeError("Singular value decomposition as optimisation problem convergence issues")
    main(args)
    