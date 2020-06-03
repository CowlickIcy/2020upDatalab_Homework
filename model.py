import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler

# %% data process
data = pd.read_csv(
    '/Users/cowlick/datasets/Facial recognition/facial-keypoints-detection/training/training.csv')  # import the training data

# print(Data.isna().sum())  # check the NAN
data.fillna(-1.0, inplace=True)  # ignore the NAN

# print(type(Data.iloc[0]['Image'])) # check the datatype
# print(Data.iloc[0]['Image']) # check the data


faces = []
facial_keypoints = []
for i in range(len(data)):
    faces.append(data['Image'][i].split(' '))
    facial_keypoints.append(data.iloc[i, :-1])

X = np.array(faces, dtype='float')  # list to array
Y = np.array(facial_keypoints, dtype='float')  # list to array
X = X.reshape(-1, 1, 96, 96)  # reshape X dimension


# mark one image
def mark(n, X, Y):
    plt.imshow(X[n, 0], cmap='gray')  # use gray channel
    facial_keypoints = int(Y.shape[1] / 2)  # point num is 15
    for i in range(facial_keypoints):
        if Y[n, i * 2] == -1:
            pass  # ignore the NAN
        else:
            plt.plot(Y[n, i * 2], Y[n, i * 2 + 1], 'o', color='r')
            plt.show()


mark(10, X, Y)  # draw the 10th image


# %% build the model

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  # (batch_size,1,96,96) to (batch_size,4,96,96)
                      out_channels=4,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.MaxPool2d(2),  # (batch_size,4,96,96) to (batch_size,4,48,48)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1),
            # (batch_size,4,48,48) to (batch_size,64,48,48)
            nn.MaxPool2d(2),  # (batch_size,64,48,48) to (batch_size,64,24,24)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # (batch_size,64,24,24) to (batch_size,128,24,24)
            nn.MaxPool2d(2),  # (batch_size,128,24,24) to (batch_size,128,12,12)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(128),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # (batch_size,128,12,12) to (batch_size,256,12,12)
            nn.MaxPool2d(2),  # (batch_size,256,12,12) to (batch_size,256,6,6)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm2d(256),
        )
        # full connection
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1024),  # (batch_size,256*6*6) to (batch_size,1024)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),  # (batch_size,1024) to (batch_size,256)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 30),  # (batch_size,256) to (batch_size,30)

        )

    def forward(self, x):
        # conv
        x = self.conv1(x)  # (batch_size,1,96,96) to (batch_size,4,48,48)
        x = self.conv2(x)  # (batch_size,4,48,48) to (batch_size,64,24,24)
        x = self.conv3(x)  # (batch_size,64,24,24) to (batch_size,128,12,12)
        x = self.conv4(x)  # (batch_size,128,12,12) to (batch_size,256,6,6)
        # make it vector
        x = x.view(x.size(0), -1)  # (batch_size,256,6,6) to (batch_size,256*6*6)
        # full connection
        x = self.fc(x)  # (batch_size,256*6*6) to (batch_size,30)
        return x


# %%  Para sets
class Parameter():
    def __init__(self):
        self.ifGPU = True
        self.epoch = 500
        self.valid2total = 1 / 5
        self.batch_size = 32
        self.learn_rate = 0.001
        self.showepoch = 25


Parm = Parameter()

if Parm.ifGPU:
    Parm.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    Parm.device = torch.device("cpu")
print(f'使用{Parm.device}')


# %%  train & test
def MSEloss_1(y, y_truth):
    y_nonan = y_truth != -1
    # calculator MSE
    loss = torch.mean((y[y_nonan] - y_truth[y_nonan]) ** 2)
    return loss


def training(model, Loss, train_loader, optimizer, Parm):
    """
    ### input
    model
    Loss
    train_loader
    optimizer
    Parm: 超参数集合
    ### output
    train_loss
    """
    # train mode
    model.train()
    # main of the training datasets
    train_loss = 0
    for batch_id, (x, y_truth) in enumerate(train_loader):
        x, y_truth = x.to(Parm.device), y_truth.to(Parm.device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = Loss(y_pred, y_truth)
        train_loss += loss.item() * x.shape[0]
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.sampler)
    return train_loss


def validing(model, Loss, valid_loader, Parm):
    """
    ### input
    model
    Loss
    valid_loader
    Parm:
    ###
    valid_loss
    """
    # test mode
    model.eval()
    # main of the test datasets
    valid_loss = 0
    for x, y_truth in valid_loader:
        x, y_truth = x.to(Parm.device), y_truth.to(Parm.device)
        y_pred = model(x)
        loss = Loss(y_pred, y_truth)
        valid_loss += loss.item() * x.shape[0]

    valid_loss /= len(valid_loader.sampler)
    return valid_loss


def train_valid_split(X, Y, Parm):
    dataset_size = len(X)
    indices = list(range(dataset_size))  # id list
    val_num = int(np.floor(Parm.valid2total * dataset_size))  # test data num

    np.random.shuffle(indices)  # shuffle id list
    train_indices, val_indices = indices[val_num:], indices[:val_num]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    loader_object = data_utils.TensorDataset(torch.from_numpy(X).float(),
                                             torch.from_numpy(Y).float())
    train_loader = data_utils.DataLoader(loader_object, batch_size=Parm.batch_size,
                                         sampler=train_sampler)
    valid_loader = data_utils.DataLoader(loader_object, batch_size=Parm.batch_size,
                                         sampler=valid_sampler)
    return train_loader, valid_loader


train_loader, valid_loader = train_valid_split(X, Y, Parm)
print(f"训练集数据总数： {len(train_loader.sampler)} | 训练集batch总数： {len(train_loader)}")
print(f"验证集数据总数： {len(valid_loader.sampler)} | 验证集batch总数： {len(valid_loader)}")

# %% start training
model = CNN()
model.to(Parm.device)

Loss = MSEloss_1  # Loss function
optimizer = optim.Adam(model.parameters(), lr=Parm.learn_rate)

train_loss = []
valid_loss = []

print("Start Training")
for epoch in range(Parm.epoch):
    # test
    train_loss.append(training(model, Loss, train_loader, optimizer, Parm))
    # verify
    valid_loss.append(validing(model, Loss, valid_loader, Parm))

    if (epoch + 1) % Parm.showepoch == 0:
        print(f"End of Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss[-1]}| Valid Loss: {valid_loss[-1]}")
print("End Training")

#  plot train_loss
plt.plot(train_loss)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
plt.show()
# plot valid_loss
plt.plot(valid_loss)
plt.title('Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('MSELoss')
plt.show()