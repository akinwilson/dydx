# numerical stability param for log of values close to 0
EPS = 1e-7

    
def flatten(v):
    for i in v:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i



def clip(x, min=EPS,max=1-EPS):
    if x.data < min:
        x.data = min
        return x
    elif x.data > max:
        x.data = max        
        return x
    else:
        return x
