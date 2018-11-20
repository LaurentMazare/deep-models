import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

batch_size = 128
epochs = 150

class ConvBN(nn.Module):
  def __init__(self, c_in, c_out):
    super(ConvBN, self).__init__()
    self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(c_out)

  def forward(self, x):
    return F.relu(self.bn(self.conv(x)))

class Residual(nn.Module):
  def __init__(self, c_in, c_out):
    super(Residual, self).__init__()
    self.pre = ConvBN(c_in, c_out)
    self.conv_bn1 = ConvBN(c_out, c_out)
    self.conv_bn2 = ConvBN(c_out, c_out)

  def forward(self, x):
    x = self.pre(x)
    x = F.max_pool2d(x, 2)
    return self.conv_bn2(self.conv_bn1(x)) + x

class ResNet9(nn.Module):
  def __init__(self, d1, d2, d3, d4):
    super(ResNet9, self).__init__()
    self.pre = ConvBN(3, d1)
    self.residual1 = Residual(d1, d2)
    self.inter = ConvBN(d2, d3)
    self.residual2 = Residual(d3, d4)
    self.linear = nn.Linear(d4, 10, bias=False)

  def forward(self, x):
    x = self.pre(x)
    x = self.residual1(x)
    x = self.inter(x)
    x = F.max_pool2d(x, 2)
    x = self.residual2(x)
    x = F.max_pool2d(x, 4)
    x = x.view(x.size(0), -1)
    x = self.linear(x)
    return x * 0.125

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda': cudnn.benchmark = True

class Cutout(object):
  def __init__(self, sz):
    self._sz = sz

  def __call__(self, img):
    h = img.size(1)
    w = img.size(2)

    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = int(np.clip(y - self._sz / 2, 0, h))
    y2 = int(np.clip(y + self._sz / 2, 0, h))
    x1 = int(np.clip(x - self._sz / 2, 0, w))
    x2 = int(np.clip(x + self._sz / 2, 0, w))
    img[:, y1:y2, x1:x2].fill_(0.0)
    return img

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Cutout(8),
    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=4)

model = ResNet9(40, 80, 160, 320)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    model.train()
    lr = 0.1 if epoch < 50 else 0.01 if epoch < 100 else 0.001
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model.forward(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()

def test(epoch):
  model.eval()
  test_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for inputs, targets in test_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model.forward(inputs)
      loss = criterion(outputs, targets)
      test_loss += loss.item() * targets.size(0)
      _, predicted = outputs.max(1)
      total += targets.size(0)
      correct += predicted.eq(targets).sum().item()
  print('%d Loss: %.4f, accuracy: %.2f%%' % (epoch, test_loss/total, 100.*correct/total))

for epoch in range(epochs):
  train(epoch)
  test(epoch)

example = torch.rand(1, 3, 32, 32).to(device)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")