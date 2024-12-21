from argparse import ArgumentParser
from dydx.metrics import Loss
from dydx.linear_algebra import Array
from dydx.layers import Linear
from dydx.autodiff import Scalar
from dydx.model import Model

def main(args):
    loss = Loss()

    # layer seed: 8433248431303025905
    # dataset seed: 3588574048688484214
    X = Array([[Scalar(0.),Scalar(0.)],
            [Scalar(0.),Scalar(1.)],
            [Scalar(1.),Scalar(0.)],
            [Scalar(1.),Scalar(1.)]])

    y  = Array([[Scalar(0.)],
                [Scalar(1.)], 
                [Scalar(1.)],
                [Scalar(0.)]])

    l1 = Linear( random=True,dims=(2, 16))
    l2 = Linear( random=True, dims=(16, 16))
    l3 = Linear( random=True,dims=(16, 1))
    layers = ['l1', 'l2', 'l3']
    model = Model(dict(zip(layers,[l1,l2,l3])))

    losses= []
    EPOCHS = 250_000_0000
    for epoch in range(args.epochs):
        # print("Training".center(70,"#"))
        y_hat = model(X)
        acc, avg_loss = loss(y,y_hat)

        avg_loss.backward()

        losses.append(avg_loss.data)
        
        info  = f'Epoch {epoch} Loss:{avg_loss.data} train accuracy:{acc*100}%'
        print(info)

        lr = 0.0000001
        for idx, p in enumerate(model.parameters()):
            p.data = p.data -  lr *p.grad
    
    
        model.zero_grad()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-ls', '--layer-seed', help='Seed to initialise computational layers')
    parser.add_argument('-e', '--epochs', help='No. of epochs', default=250_000_0000)
    parser.add_argument('-lr', '--learning-rate', help='Step size for gradient descent algorithm', default=0.0000001)
    
    args = parser.parse_args()
    print('args.epochs', args.epochs)
    print('args.layer_seed', args.layer_seed)
    print('args.learning_rate', args.learning_rate)
    
    