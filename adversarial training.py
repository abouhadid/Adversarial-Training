import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
results = []

total = 0
correct = 0
for epoch in range(10):
    print("epoch :" + str(epoch))
    total = 0
    correct = 0
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_images = fgsm_attack(net, criterion, inputs, targets, 0.031)

        concatenated_images = torch.cat((adv_images, inputs), 0)
        concatenated_targets = torch.cat((targets, targets), 0)

        outputs = net(adv_images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(total_loss)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_images = fgsm_attack(net, criterion, inputs, targets, 0.031)
        outputs = net(adv_images)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('accuracy on adversarial testset:' + str(100 * correct / total))

    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('accuracy on original testset:' + str(100 * correct / total))

