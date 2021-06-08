import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import build_mnist_loader
from models import *
from nn_layers import *

import argparse
import importlib
import os
import time
# optionally import SummaryWriter for tensorboard logging
if importlib.util.find_spec('tensorboard') is not None:
    from torch.utils.tensorboard import SummaryWriter
    TB_WRITER = SummaryWriter()
else:
    TB_WRITER = None


MODELS = ['cnn_same_channels',
          'cnn_same_params',
          'mobius_scalar',
          'mobius_signflip',
          'mobius_regular',
          'mobius_irrep',
          'mobius_mixed']


class Logger(object):
    """Simple logger class that creates a log file and mirrors all inputs to write to the file and stdout"""
    def __init__(self, arguments):
        os.makedirs('logs', exist_ok=True)
        logfile = '{}_{}|{}_{}.log'.format(arguments['model'], arguments['train_mode'], arguments['test_mode'],
                                                   time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime()))
        self.logfile = os.path.join('logs', logfile)
        self.write('training arguments:')
        for k,v in arguments.items():
            self.write('\t{:s} = {:s}'.format(k,str(v)))

    def write(self, string, print_bool=True, **print_kwargs):
        if print_bool:
            print(string, **print_kwargs)
        with open(self.logfile, 'a') as f:
            f.write('\n'+string)

    def prepend(self, string, print_bool=True, **print_kwargs): # for convenience to prepend final results
        if print_bool:
            print(string, **print_kwargs)
        with open(self.logfile, 'r') as original:
            data = original.read()
        with open(self.logfile, 'w') as modified:
            modified.write(string + '\n' + data)


def accuracy(logits, labels):
    predictions = logits.argmax(dim=1)
    predictions = predictions.to(dtype=labels.dtype)
    accuracy = float((labels == predictions).sum()) / predictions.numel()
    return accuracy


def train_loop(train_loader, model, device, optimizer, epoch, num_epochs, global_step):
    model.train()
    losses = []
    accs = []
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(logits.detach(), labels.detach())
        losses.append(loss.item())
        accs.append(acc)

        if TB_WRITER is not None:
            TB_WRITER.add_scalar('loss_train', loss.item(), global_step)
            TB_WRITER.add_scalar('acc_train', acc, global_step)
        print('Epoch [{}/{}], Iter [{}/{}] Loss: {:.4f}, Acc:{:5.2f}'.format(
                         epoch, num_epochs, i+1, len(train_loader), loss.item(), 100*acc))
        global_step += 1
    avg_loss = np.mean(losses)
    avg_acc = np.mean(accs)
    print('Epoch [{}/{}]: AvgLoss: {:.4f}, AvgAcc:{:5.2f}'.format(epoch, num_epochs, avg_loss, 100*avg_acc))
    return avg_loss, avg_acc, global_step


def evaluate(test_loader, model, device):
    correct = 0
    total = 0
    model.eval()
    for images, labels in test_loader:
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).long().sum().item()
    eval_acc = 100*correct/total
    return eval_acc


def train_model(model, num_epochs, weight_decay, bs,
                lr, lr_decay_period, lr_decay_factor,
                eval_freq, train_mode, test_mode):
    logger = Logger(locals())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # instantiate model
    if model == 'cnn_same_channels':
        model = CNN(fix_params=False)
    if model == 'cnn_same_params':
        model = CNN(fix_params=True)
    if model == 'mobius_scalar':
        model = MobiusGaugeCNN(mode='scalar')
    if model == 'mobius_signflip':
        model = MobiusGaugeCNN(mode='signflip')
    if model == 'mobius_regular':
        model = MobiusGaugeCNN(mode='regular')
    if model == 'mobius_irrep':
        model = MobiusGaugeCNN(mode='irrep')
    if model == 'mobius_mixed':
        model = MobiusGaugeCNN(mode='mixed')
    model.to(device)
    N_params = sum([param.flatten().shape[0] for param in model.parameters()])
    logger.write('N_params = {}'.format(N_params))

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    train_loader = build_mnist_loader('train', shifted=(train_mode=='shifted'), batch_size=bs, num_workers=4)
    test_loaders = []
    if test_mode in ('shifted', 'both'):
        test_loaders.append(build_mnist_loader('test', shifted=True, batch_size=bs, num_workers=4))
    if test_mode in ('centered', 'both'):
        test_loaders.append(build_mnist_loader('test', shifted=False, batch_size=bs, num_workers=4))

    global_step = 0
    for epoch in range(1,1+num_epochs):
        # train
        _, avg_train_acc, global_step = train_loop(train_loader, model, device,
                                                   optimizer, epoch, num_epochs, global_step)
        # validate (on test set, as common on MNIST)
        if eval_freq!=0:
            if (epoch%eval_freq)==0 and epoch!=num_epochs:
                test_accs = [evaluate(tl, model, device) for tl in test_loaders]
                logger.write('eval accuracy = {}'.format(test_accs))
                if TB_WRITER is not None:
                    epoch_accs = {'train': avg_train_acc}
                    if test_mode == 'both':
                        epoch_accs['shifted'] = test_accs[0]
                        epoch_accs['centered'] = test_accs[1]
                    TB_WRITER.add_scalars('epoch_accuracies', epoch_accs, epoch)
        # decay learning rate
        if lr_decay_period!=0:
            if epoch%lr_decay_period == 0:
                lr /= lr_decay_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                logger.write('updated learning rate to {}'.format(lr))

    # test on full test set
    test_accs = [evaluate(tl, model, device) for tl in test_loaders]
    logger.prepend('final test accuracy = {}'.format(test_accs))

    return test_accs



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='The model to be trained.',
                        type=str,
                        choices=MODELS,
                        required=True)
    parser.add_argument('--train_mode',
                        help='Select whether the digits in the train set are uniformly shifted along the Mobius strip \
                            or centered at one location/orientation.',
                        type=str,
                        default='centered',
                        choices=['shifted', 'centered'])
    parser.add_argument('--test_mode',
                        help='Select whether the digits in the test set are uniformly shifted along the Mobius strip, \
                            centered at one location/orientation or both are evaluated.',
                        type=str,
                        default='both',
                        choices=['shifted', 'centered', 'both'])
    parser.add_argument('--num_epochs',
                        help='Number of training epochs.',
                        type=int,
                        default=20)
    parser.add_argument('--batch_size',
                        help='Number of samples per iteration.',
                        type=int,
                        default=128)
    parser.add_argument('--weight_decay',
                        help='L2 regularization on all model parameters.',
                        type=float,
                        default=1e-6)
    parser.add_argument('--lr',
                        help='Learning rate.',
                        type=float,
                        default=5e-3)
    parser.add_argument('--lr_decay_period',
                        help='Number of epochs after which the lr is being decayed.',
                        type=int,
                        default=4)
    parser.add_argument('--lr_decay_factor',
                        help='Factor by which the learning rate is being decayed after every lr_decay_period epochs.',
                        type=float,
                        default=2.)
    parser.add_argument('--eval_freq',
                        help='Evaluation frequency in epochs. Zero means no intermediate evaluation.',
                        type=int,
                        default=1)
    args = parser.parse_args()

    train_model(model=args.model,
         num_epochs=args.num_epochs, weight_decay=args.weight_decay, bs=args.batch_size,
         lr=args.lr, lr_decay_period=args.lr_decay_period, lr_decay_factor=args.lr_decay_factor,
         eval_freq=args.eval_freq, train_mode=args.train_mode, test_mode=args.test_mode)


