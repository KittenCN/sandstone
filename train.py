import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from model import VGG11, resnet50
from tqdm import tqdm
from torchvision import transforms

use_gpu = torch.cuda.is_available()
pklfile = r'./model.pth'

transform = transforms.Compose([
    transforms.ToTensor(),
])

X_train = np.array(np.load('X_train_split.npy'))
y_train = np.array(np.load('y_train_split.npy'))
X_test = np.array(np.load('x_test_split.npy'))
y_test = np.array(np.load('y_test_split.npy'))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)

net = VGG11()
if os.path.exists(pklfile):
        net.load_state_dict(torch.load(pklfile))
        net.eval()
        print("load old model")
else:
    print("creat new model")
if use_gpu:
    net = net.cuda()
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=1e-3)

epochs = 10
pbar = tqdm(total=epochs * len(train_loader))

for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                _, watched = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == watched).sum().item()

        pbar.update(1)
        pbar.set_description("train loss: %e, test acc: %e" % (loss.item(), correct / total))

        # if (i + 1) % 10 == 0:
        #     print('[%d, %5d] loss: %e' % (epoch + 1, i + 1, loss.item()))
    if (epoch + 1) % 10 == 0:
        # print('epoch: %d, loss: %e' % (epoch + 1, loss.item()))
        torch.save(net.state_dict(), 'model.pth')

pbar.close()
print('Finished Training')
print('Final loss: %e' % loss.item())

# run test dataset
correct = 0
total = 0
pbar = tqdm(total=len(test_loader))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        _, watched = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == watched).sum().item()
        pbar.update(1)
pbar.close()
print('Accuracy of the network: %d %%' % (100 * correct / total))
