##############################################################################################################
# This section imports the required libraries
#
#
##############################################################################################################
import os
import numpy as np
from pathlib import Path
from data_prep_bbh import *
from utils import *

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn

##############################################################################################################
# This section defines the data generator
#
#
##############################################################################################################


class DatasetGenerator(Dataset):
    def __init__(self, fs=8192, T=1, snr=20,
                 detectors=['H1', 'L1'],
                 nsample_perepoch=100,
                 Nnoise=25, mdist='metric',beta=[0.75,0.95],
                 verbose=True):
        # Initialization function, set various parameters
        if verbose:
            print('GPU available?', torch.cuda.is_available())
        self.fs = fs     # the sampling frequency (Hz)
        self.T = T       # the observation duration (sec)

        safe = 2         # define the safe multiplication scale for the desired time length
        self.T *= safe

        self.detectors = detectors
        self.snr = snr
        
        self.generate(nsample_perepoch, Nnoise, mdist, beta)  # pre-generate samples

    def generate(self, Nblock, Nnoise=25, mdist='metric',beta=[0.75,0.95]):
        # Function to generate data
        # Nnoise: # the number of noise realizations per signal
        # Nblock: # the number of training samples per output file
        # mdist:  # mass distribution (astro,gh,metric)

        ts, par = sim_data(self.fs, self.T, self.snr, self.detectors, Nnoise, size=Nblock, mdist=mdist,
                           beta=beta, verbose=False)
        self.strains = np.expand_dims(ts[0], 1)   # (nsample, 1, len(det), fs*T)
        self.labels = ts[1]

    def __len__(self):
        # Return the length of the dataset
        return len(self.strains)

    def __getitem__(self, idx):
        # Function to get a data sample
        return self.strains[idx], self.labels[idx]


##############################################################################################################
# This section defines the network structure, and the functions to load and save the model
#
#
##############################################################################################################

# In the model definition, we define a convolutional neural network with multiple convolutional layers, 
# activation functions, batch normalization layers, and max-pooling layers.
# Finally, a Flatten layer and two fully connected layers are added.

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # ������ (1, 3) ȷ��ֻ��ʱ��ά��(8192)�Ͼ�������ѹ��̽����ά��(2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        
        # ���� ResNet-18 �Ľṹ����
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)
        
        # ����Ӧ�ػ���ά��ѹƽΪ 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# In the model save/load functions, we save model parameters, optimizer state, scheduler state, and training epoch.
# When loading the model, we load the model parameters and return the model, epoch, and training loss history.


def load_model(checkpoint_dir=None):
    # Function to load model
    net = MyResNet()

    if (checkpoint_dir is not None) and (Path(checkpoint_dir).is_dir()):
        p = Path(checkpoint_dir)
        files = [f for f in os.listdir(p) if '.pt' in f]

        # if there is a *.pt model file, load it!
        if (files != []) and (len(files) == 1):
            checkpoint = torch.load(p / files[0])
            net.load_state_dict(checkpoint['model_state_dict'])
        print('Load network from', p / files[0])
        
        epoch = checkpoint['epoch']
        train_loss_history = np.load(p / 'train_loss_history_cnn.npy').tolist()
        return net, epoch, train_loss_history
    else:
        print('Init network!')
        return net, 0, []


def save_model(epoch, model, optimizer, scheduler, checkpoint_dir, train_loss_history, filename):
    """Save a model and optimizer to file.
    """
    # Function to save model
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)

    # clear all the *.pt files
    assert '.pt' in filename
    for f in [f for f in os.listdir(p) if '.pt' in f]:
        os.remove(p / f)

    # Save loss history
    np.save(p / 'train_loss_history_cnn', train_loss_history)

    output = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }

    if scheduler is not None:
        output['scheduler_state_dict'] = scheduler.state_dict()
    # save the model
    torch.save(output, p / filename)


##############################################################################################################
# This section defines the training and evaluation functions
#
#
##############################################################################################################

numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    # Compute number of correct predictions
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, dim=1)        
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, loss_func, device=None): #@save
    """Compute model accuracy on a dataset using GPU"""
    if isinstance(net, nn.Module):
        net.eval()  # Set to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # Number of correct predictions, total samples, test loss
    metric = Accumulator(3)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device).to(torch.float)
            y = y.to(device).to(torch.long)
            y_hat = net(X)
            loss = loss_func(y_hat, y)
            metric.add(accuracy(y_hat, y), y.numel(), loss.sum())
    return metric[0] / metric[1], metric[2] / metric[1]


# In the training function, we first define the loss function, optimizer, and learning rate scheduler.
# Then we start the training loop. For each epoch, new training samples are generated,
# forward propagation is performed, the loss is computed, and backpropagation updates the parameters.
# At the end of each epoch, the model is evaluated on the test set, and the model with the lowest test loss is saved.


def train(net, lr, nsample_perepoch, epoch, total_epochs,
          dataset_train, data_loader, test_iter,
          train_loss_history, checkpoint_dir, device, notebook=True):
    """Training function"""
    # Set optimizer parameters
    loss_func = nn.CrossEntropyLoss()  # Define loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # Define optimizer
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_epochs,  # Define LR scheduler
                    )

    torch.cuda.empty_cache()  # Clear CUDA cache
    if notebook:
        animator = Animator(xlabel='epoch', xlim=[1, total_epochs],
                                legend=['train loss', 'test loss', 'train acc', 'test acc'])  # Define animator
    timer, num_batches = Timer(), len(dataset_train)  # Define timer and batch count

    # Start training loop
    for epoch in range(epoch, epoch + total_epochs):
        # Pre-generate training samples
        dataset_train.generate(nsample_perepoch)

        # Print current learning rate if not in notebook mode
        if not notebook:
            print('Learning rate: {}'.format(
                        optimizer.state_dict()['param_groups'][0]['lr']))

        # Initialize metrics accumulator: sum of training loss, sum of training accuracy, sample count
        metric = Accumulator(3)

        # Set network to training mode
        net.train()
        # Iterate over all batches in data loader
        for batch_idx, (x, y) in enumerate(data_loader):
            # Start timer
            timer.start()
            # Zero optimizer gradients (1/4)
            optimizer.zero_grad()

            # Move data and labels to device and convert to appropriate types
            data = x.to(device, non_blocking=True).to(torch.float)
            label = y.to(device, non_blocking=True).to(torch.long)

            # Forward pass
            pred = net(data)
            # Compute loss
            loss = loss_func(pred, label)

            # Update metrics without computing gradients
            with torch.no_grad():
                # Update metrics accumulator
                metric.add(loss.sum(), accuracy(pred, label), x.shape[0])
            # Stop timer
            timer.stop()

            # Backward pass to compute gradients (2/4)
            loss.backward()
            # Update network parameters using optimizer (3/4)
            optimizer.step()

            # Compute training loss and accuracy
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # Update animator if in notebook mode and at 1/5 of batches or last batch
            if notebook and (batch_idx + 1) % (num_batches // 5) == 0 or batch_idx == num_batches - 1:
                animator.add(epoch + (batch_idx + 1) / num_batches,
                             (train_l, None, train_acc, None))

        # Update learning rate using scheduler (4/4)
        scheduler.step()

        # Evaluate model on test set
        test_acc, test_l = evaluate_accuracy_gpu(net, test_iter, loss_func, device)

        # Save training loss history
        train_loss_history.append([epoch+1, train_l, test_l, train_acc, test_acc])

        # Update animator if notebook mode; otherwise, print train/test loss and accuracy
        if notebook:
            animator.add(epoch + 1, (train_l, test_l, train_acc, test_acc))
        else:
            print(f'Epoch: {epoch+1} \t'
                  f'Train Loss: {train_l:.4f} Test Loss: {test_l:.4f} \t'
                  f'Train Acc: {train_acc} Test Acc: {test_acc}')

        # Save model if current test loss is the lowest so far
        if (test_l <= min(np.asarray(train_loss_history)[:,1])):
            save_model(epoch, net, optimizer, scheduler, 
                       checkpoint_dir=checkpoint_dir,
                       train_loss_history=train_loss_history,
                       filename=f'model_e{epoch}.pt',)

    # Print final training loss, training accuracy, and test accuracy
    print(f'loss {train_l:.4f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    # Print examples processed per second and device used
    print(f'{metric[2] * total_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == "__main__":
    # Main function, entry point of the program
    nsample_perepoch = 100  # Number of samples per epoch
    dataset_train = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)  # Training dataset
    dataset_test = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)  # Test dataset

    # Create a DataLoader
    data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True,)  # Training data loader
    test_iter = DataLoader(dataset_test, batch_size=32, shuffle=True,)  # Test data loader

    device = torch.device('cuda')  # Use CUDA device

    # Path to save model and training loss history
    checkpoint_dir = './checkpoints_cnn1/'

    # Create model    
    net, epoch, train_loss_history = load_model(checkpoint_dir)  # Load model
    net.to(device);  # Move model to device

    # Optimizer parameters
    lr = 0.003  # Learning rate
    total_epochs = 100  # Total number of epochs
    total_epochs += epoch  # Add already trained epochs
    output_freq = 1  # Output frequency

    train(net, lr, nsample_perepoch, epoch, total_epochs, data_loader, test_iter, notebook=False)  # Train model