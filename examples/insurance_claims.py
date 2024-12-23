from argparse import ArgumentParser
from dydx.layers import Linear, Embedding
from dydx.model import Model
from dydx.download import encodings_dims, encodings_dims
from dydx.metrics import Loss
from dydx.dataset import Dataloader, Dataset
from pathlib import Path
from functools import reduce
from datetime import datetime
import math
import yaml


def propagate(x,model):    
    a_in = x[:,1]
    x0 = model.e0(a_in)
    # Destination
    d_in = x[:,4]
    x1 = model.e1(d_in)
    # Gender
    g_in = x[:,7]
    x2 = model.e2(g_in)
    # Product Name
    p_in = x[:,9]
    x3 = model.e3(p_in)
    x4 = x[:,0]
    x5 = x[:,2]
    x6 = x[:,3]
    x7 = x[:,5]
    x8 = x[:,6]
    x9 = x[:,8]
    features = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
    stack = lambda x, y : x.stack(y,dim=0)
    x = reduce(stack, features)
    x = model.l1(x)
    x = model.l2(x)
    x = model.l3(x)
    x = model.l4(x)
    y_hat = model.l5(x)
    
    return y_hat, model


def main(args):
    cat_data_size  = {k:v for (k,v) in encodings_dims.items() if v > 2 }

    layer_seeds = args.layer_seeds 
    
    ls_embedders = layer_seeds[:len(cat_data_size.values())]
    ls_linear = layer_seeds[len(cat_data_size.values()):]
    
    embedders =[ (f'e{idx}',Embedding(dims=(in_size, args.embedding_dim),seed=s)) for (idx,(in_size,s)) in  enumerate(zip(cat_data_size.values(), ls_embedders) )]


    l1 = Linear(dims=(4*args.embedding_dim + 6, args.scaler*64),seed=ls_linear[0] )
    l2 = Linear(dims=(args.scaler*64, args.scaler*64),seed=ls_linear[1])
    l3 = Linear(dims=(args.scaler*64, args.scaler*32),seed=ls_linear[2])
    l4 = Linear(dims=(args.scaler*32, 16), seed=ls_linear[3])
    l5 = Linear(activation=False, dims=(16,1), seed=ls_linear[4])
    # when activation is set to false we use sigmoid function to normalised output 

            
    names = ["l1","l2","l3","l4", "l5"]		
    layers =[l1,l2,l3,l4, l5]
    ls = {**dict(embedders), **dict(zip(names,layers))}
    model = Model(ls)
    
    foldername = datetime.now().strftime('%d_%m_%Y.%H:%M:%S')
    fname = "fitting_history.data"
    pname = "params.yaml"

    
    exp_root = Path(__file__).parent.parent / "experiments" / foldername 
    exp_root.mkdir(parents=True, exist_ok=True)
    fp_fitting  = exp_root / fname
    fp_params = exp_root / pname

    loss = Loss()
    ds = Dataset(testing=False, logging=True) # takes 4048 examples to overfit model too for testing purposes
    
    splits = ['train','val','test']
    print(f"dataset sizes for {', '.join(splits)} are {[ds.__len__(split) for split in splits]} respectively.")
    dst = ds.train
    
    # dsv = ds.val
    # dss = ds.test
    
    # iterators 
    dltrain  = Dataloader(dst,args.batch_size)
    losses = []
    PARMAS = { 'initialisation': {'layer_seeds': model.seeds(), 
                                'dataset_seed': ds.seed },
            
            'fitting':{'epochs': args.epochs ,'learning_rate':args.learning_rate, 'optimiser': 
                args.optimisation},
            'model': {'param_count': len(model.parameters())} }
            
    with open(fp_params, "w") as f:
        yaml.dump(PARMAS, f)
            
    delta_t_minus1 = [0] *  len(model.parameters())
    step = 0 
    for epoch in range(args.epochs):
        # print("Training".center(70,"#"))
        for (idx,xy) in enumerate( dltrain()):
            x,y = xy
            
            # y_hat =model(x)		
            y_hat, model = propagate(x,model)
            acc, avg_loss = loss(y,y_hat)

            avg_loss.backward()

            losses.append(avg_loss.data)
            
            info  = f'Step {step} Loss:{avg_loss.data} train accuracy:{acc*100}%'
            with open(fp_fitting, "a+") as f:
                f.write(info + '\n')
    
            print(info)
            # optimizer step
            if args.optimisation == 'sgd+m':
                # SGD with momentum
                for idx, p in enumerate(model.parameters()):
                    p.data = p.data -  args.learning_rate *( args.beta *p.grad + (1-args.beta)*delta_t_minus1[idx])
                    delta_t_minus1[idx] =  p.grad + (1-args.beta)*delta_t_minus1[idx]
            if args.optimisation == 'sgd':
                # SGD 
                for idx, p in enumerate(model.parameters()):
                    p.data = p.data -   args.learning_rate *p.grad
            model.zero_grad()
            step += 1
            if math.isnan(avg_loss.data):
                print('Cutting routine, loss return was NotANumber; nan')
                


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('-s', '--layer-scale', help='parameter controls the size of the hidden layers', default=4)
    parser.add_argument('-ds', '--dataset-seed', help='seed value for initialisation of dataset', default=4051899315611228202)
    parser.add_argument('-ls' '--layer-seeds', help='seed values for initialisation of layers', default='2657719702506702394, 3451391271922671329, 4049295135587188773, 8630703877296012607, 5626589656863341942, 5626589656863341942, 5626589656863341942, 5626589656863341942, 5626589656863341942')
    parser.add_argument('-ed', '--embedding-dim', help='dimension of embeddings for categorical data', default=32)
    parser.add_argument('-lr', '--learning-rate', help='step size during optimisation step', default=0.0000001)
    parser.add_argument('-e', '--epochs', help='number of epochs to run algorithm for', default=1000)
    parser.add_argument('-o', '--optimisation', help='optimisation algorithm to use. Can be either "sgd" or  "sgd+m" ', default='sgd+m')
    parser.add_argument('-b', '--beta', help='momentum parameter for sgd+m optimisation algorithm', default=0.9)
    parser.add_argument('-bs', '--batch-size', help="batch size during fitting", default=32)
    
    args = parser.parse_args()
    args.layer_seeds = [int(x) for x in args.layer_seeds.split(',')]
    
    main(args)
    