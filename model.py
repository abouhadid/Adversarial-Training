#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from white_box.fgsm import  FGSM
from white_box.pgd import  PGD
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

batch_size = 32


class Net(nn.Module):
    model_file = "models/default_model.pth"

    def __init__(self):
        super(Net, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

    def load_for_testing(self, project_dir='./'):
        self.load(os.path.join(project_dir, Net.model_file))


def train_model(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print statistics
            running_loss += loss.item()
        acc = 100 * correct / total
        print("Loss :" + str(running_loss))
        print("Accuracy :" + str(acc))

    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))


def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total
def test_adv(net, test_loader):
    '''Basic testing function.'''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    fgsm=FGSM(net,criterion)
    pgd=PGD(net,criterion)

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    
    for i, data in enumerate(test_loader):
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        adv_images_fgsm=fgsm.attack(images,labels)
        adv_images_pgd=pgd.attack(images,labels)

        concatenated_images = torch.cat((adv_images_fgsm,adv_images_pgd), 0)
        concatenated_targets = torch.cat((labels, labels), 0)
        outputs = net(concatenated_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == concatenated_targets).sum().item()
    return 100 * correct / total
def adv_trainning(net, train_loader, pth_filename, num_epochs):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    fgsm=FGSM(net,criterion)
    pgd=PGD(net,criterion)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print("\n\nEpoch :"+str(epoch))
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            adv_images_fgsm=fgsm.attack(inputs,labels)
            if i ==0:print("FGSM images generated...")
            adv_images_pgd=fgsm.attack(inputs,labels)
            if i==0:print("PGD images generated...")
            concatenated_images = torch.cat((adv_images_fgsm,adv_images_pgd, inputs), 0)
            concatenated_targets = torch.cat((labels, labels,labels), 0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(concatenated_images)
            loss = criterion(outputs, concatenated_targets)
            loss.backward()
            optimizer.step()


    net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))


def get_train_loader(batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader


def get_validation_loader(batch_size=32):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''
    transform_test = transforms.Compose([
        transforms.ToTensor(),

    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return testloader


def main():
    #### Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=VGG.model_file,
                        help="Name of the file used to load or to sore the model weights." \
                             "If the file exists, the weights will be load from it." \
                             "If the file doesn't exists, or if --force-train is set, training will be performed, " \
                             "and the model weights will be stored in this file." \
                             "Warning: " + VGG.model_file + " will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists" \
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")

    parser.add_argument('-a', '--adv-train', action="store_true",
                        help="Start the adversarial training. It requires that the original model must be stored in models directory")
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)

        train_loader = get_train_loader(batch_size=32)
        train_model(net, train_loader, args.model_file, args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    #### Model adversarial training
    if os.path.exists(args.model_file) and args.adv_train :
        print("Adversarial Training")
        net.load(args.model_file)
        train_loader = get_train_loader(batch_size=32)
        adv_trainning(net, train_loader, args.model_file, args.num_epochs)

 

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.

    valid_loader = get_validation_loader()

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))
    acc = test_adv(net, valid_loader)
    print("Model adverarial accuracy (valid): {}".format(acc))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, " \
              "it will not be the one used for testing your project. " \
              "If this is your best model, " \
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))


if __name__ == "__main__":
    main()
