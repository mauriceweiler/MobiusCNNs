import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_layers import *


class MobiusGaugeCNN(nn.Module):
    """Implementation of the orientation independent CNNs on the Mobius strip.
    
    An overview of the models is given in Table 4 of the paper.
    """
    
    def __init__(self, kernel_size=5, ch_in=1, classes=10, mode='regular'):
        """
        Args:
            kernel_size (int): Size of the convolution kernel in pixels. Defaults to 5.
            ch_in (int): Number of input scalar fields. Defaults to 1.
            classes (int): Number of classes. Defaults to 10.
            mode (str): Type of the model, controlling the multiplicities of feature fields. Different modes result in
                approximately the same number of channels and parameters. Defaults to 'regular'.

        """
        super(MobiusGaugeCNN, self).__init__()
        assert mode in ('scalar', 'signflip', 'regular', 'irrep', 'mixed')
        if mode == 'scalar':
            multipl = np.array([4,0,0])
        if mode == 'signflip':
            multipl = np.array([0,4,0])
        if mode == 'regular':
            multipl = np.array([0,0,2])
        if mode == 'irrep':
            multipl = np.array([2,2,0])
        if mode == 'mixed':
            multipl = np.array([1,1,1])

        self.mobius_cnn = nn.Sequential(
            # the first layer takes the ch_in input scalar field and maps it to the fields as specified by mode
            MobiusConv(in_fields=(ch_in,0,0), out_fields=4*multipl, kernel_size=kernel_size),
            EquivNonlin(in_fields=4*multipl),
            #
            MobiusConv(in_fields=4*multipl, out_fields=8*multipl, kernel_size=kernel_size),
            EquivNonlin(in_fields=8*multipl),
            # POOL 28->14 #####################################################################
            MobiusPool(in_fields=8*multipl), # 28-->14 px
            MobiusConv(in_fields=8*multipl, out_fields=16*multipl, kernel_size=kernel_size),
            EquivNonlin(in_fields=16*multipl),
            # 
            MobiusConv(in_fields=16*multipl, out_fields=32*multipl, kernel_size=kernel_size),
            EquivNonlin(in_fields=32*multipl),
            # POOL 14->7 #####################################################################
            MobiusPool(in_fields=32*multipl), # 14-->7 px
            MobiusConv(in_fields=32*multipl, out_fields=64*multipl, kernel_size=kernel_size),
            EquivNonlin(in_fields=64*multipl),
            # the final layer maps to scalar fields to produce gauge invariant predictions
            MobiusConv(in_fields=64*multipl, out_fields=(64,0,0), kernel_size=kernel_size),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=64),
            nn.ELU(),
            nn.Dropout(p=.3),
            nn.Linear(64, 32),
            #
            nn.BatchNorm1d(num_features=32),
            nn.ELU(),
            nn.Dropout(p=.3),
            nn.Linear(32, classes)
        )

    def forward(self, input: torch.Tensor):
        """Runs gauge equivariant convolution on Mobius strip"""
        features = self.mobius_cnn(input)
        # pool resulting fields to obtain position invariant features
        # the use of F.max_pool2d is valid since self.mobius_cnn returns only scalar fields
        # the resulting features will therefore be both gauge and position invariant
        N,C,W,L = features.shape
        features = F.max_pool2d(features, kernel_size=(W,L)).view(N,C)
        # classify based on gauge invariant features
        features = self.classifier(features)
        return features



class CNN(nn.Module):
    """Conventional CNN baselines, which are *not* orientation independent"""
    def __init__(self, kernel_size=5, ch_in=1, classes=10, fix_params=False):
        """
        Args:
            kernel_size (int): Size of the convolution kernel in pixels. Defaults to 5.
            ch_in (int): Number of input scalar fields. Defaults to 1.
            classes (int): Number of classes. Defaults to 10.
            fix_params (bool): Determines whether the number of parameters is fixed to that of the orientation
                independent CNNs or not. Since non-equivariant models are less parameter efficient, this requires a
                factor of sqrt(2) less channels. If set to False, the model has the same number of channels like the
                orientation independent models but has more parameters. Defaults to False.

        """
        super(CNN, self).__init__()
        self.kernel_size = kernel_size
        self.fix_params = fix_params
        fix_params = (np.sqrt(2) if fix_params else 1) # approximately fix number of parameters, by default keep same number of channels

        self.cnn = nn.Sequential(
            MobiusPadNaive(kernel_size),
            nn.Conv2d(in_channels=ch_in, out_channels=int(16//fix_params), kernel_size=kernel_size),
            nn.ELU(),
            #
            MobiusPadNaive(kernel_size),
            nn.Conv2d(in_channels=int(16//fix_params), out_channels=int(32//fix_params), kernel_size=kernel_size),
            nn.ELU(),
            # POOL 28->14 #####################################################################
            MobiusPool(in_fields=(int(32//fix_params),0,0)), # 28-->14 px
            MobiusPadNaive(kernel_size),
            nn.Conv2d(in_channels=int(32//fix_params), out_channels=int(64//fix_params), kernel_size=kernel_size),
            nn.ELU(),
            # 
            MobiusPadNaive(kernel_size),
            nn.Conv2d(in_channels=int(64//fix_params), out_channels=int(128//fix_params), kernel_size=kernel_size),
            nn.ELU(),
            # POOL 14->7 #####################################################################
            MobiusPool(in_fields=(int(128//fix_params),0,0)), # 14-->7 px
            MobiusPadNaive(kernel_size),
            nn.Conv2d(in_channels=int(128//fix_params), out_channels=int(256//fix_params), kernel_size=kernel_size),
            nn.ELU(),
            #
            MobiusPadNaive(kernel_size),
            nn.Conv2d(in_channels=int(256//fix_params), out_channels=64, kernel_size=kernel_size),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features=64),
            nn.ELU(),
            nn.Dropout(p=.3),
            nn.Linear(64, 32),
            #
            nn.BatchNorm1d(num_features=32),
            nn.ELU(),
            nn.Dropout(p=.3),
            nn.Linear(32, classes)
        )

    def forward(self, input: torch.Tensor):
        # run non-equivariant (orientation dependent) convolution on Mobius strip
        features = self.cnn(input)
        # pool resulting feature fields - the prediction is gauge dependent
        N,C,W,L = features.shape
        features = F.max_pool2d(features, (W,L)).view(N,C)
        # classify
        features = self.classifier(features)
        return features





if __name__ == '__main__':

    # count number of parameters of the models
    def count_params(model):
        param_shapes = [p.shape for p in model.parameters()]
        return np.sum([np.prod(shape) for shape in param_shapes])

    model_cnn            = CNN(fix_params=False)
    model_cnn_fix_params = CNN(fix_params=True)
    model_scalar  = MobiusGaugeCNN(mode='scalar')
    model_sign    = MobiusGaugeCNN(mode='signflip')
    model_regular = MobiusGaugeCNN(mode='regular')
    model_irrep   = MobiusGaugeCNN(mode='irrep')
    model_mixed   = MobiusGaugeCNN(mode='mixed')

    print('count number of parameters for individual models:')
    print('  params={:7}, model=cnn_channels'.format(count_params(model_cnn)))
    print('  params={:7}, model=cnn_params'.format(count_params(model_cnn_fix_params)))
    print('  params={:7}, model=scalar'.format(count_params(model_scalar)))
    print('  params={:7}, model=signflip'.format(count_params(model_sign)))
    print('  params={:7}, model=regular'.format(count_params(model_regular)))
    print('  params={:7}, model=irrep'.format(count_params(model_irrep)))
    print('  params={:7}, model=mixed'.format(count_params(model_mixed)))
