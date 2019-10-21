import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, layer_size, kernel_size, conv_num, batch_norm):
        super(Net, self).__init__()
        self.layer_size = layer_size
        self.kernel_size = kernel_size
        self.conv_num = conv_num
        self.batch_norm = batch_norm
        self.pool = nn.MaxPool2d(2, 2)
        #4 convolution layers
        self.conv1 = nn.Conv2d(3, self.kernel_size[0], 7) # 56 -> 50c1 -> 46c2 ->42c3-> 21 ->17 ->8.5 -> 7 -> 3
        self.conv2 = nn.Conv2d(self.kernel_size[0], self.kernel_size[1], 5) #56 -> conv1 -> 50 -> conv2 46 -> conv3 -> 42 -> pool -> 21 - > conv4 -> 17 -> pool -> 8 -> conv5 -> 7 -> pool -> 3
        self.conv3 = nn.Conv2d(self.kernel_size[1], self.kernel_size[2], 5)
        self.conv4 = nn.Conv2d(self.kernel_size[2], self.kernel_size[3], 5)
        self.conv5 = nn.Conv2d(self.kernel_size[3], self.kernel_size[4], 2)


        #for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
         #   nn.init.xavier_uniform(conv.weight)

        # 4 batch normalization layers
        self.bn1 = nn.BatchNorm2d(self.kernel_size[0])
        self.bn2 = nn.BatchNorm2d(self.kernel_size[1])
        self.bn3 = nn.BatchNorm2d(self.kernel_size[2])
        self.bn4 = nn.BatchNorm2d(self.kernel_size[3])
        self.bn5 = nn.BatchNorm2d(self.kernel_size[4])
        # 2 layers
        if self.conv_num == 2:
            self.fc1 = nn.Linear(self.kernel_size[3] * 11 * 11, 100)
        if self.conv_num == 4:
            self.fc1 = nn.Linear(self.kernel_size[4] * 8 * 8, 100)
        self.fc2 = nn.Linear(100, self.layer_size)
        # 2 normalized layers
        self.fc1_bn = nn.BatchNorm1d(self.layer_size)
        self.fc2_bn = nn.BatchNorm1d(self.layer_size)
        #output layer
        self.fc3 = nn.Linear(self.layer_size, 10)

    def forward(self, x):
        if (self.conv_num ==2 and self.batch_norm == False):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, self.kernel_size[3]* 11 *  11)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        if (self.conv_num == 2 and self.batch_norm == True):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, self.kernel_size[3] * 11 * 11)
            x = F.relu(self.fc1_bn(self.fc1(x)))
            x = F.relu(self.fc2_bn(self.fc2(x)))

            x = self.fc3(x)
        if (self.conv_num == 4 and self.batch_norm == False):
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = x.view(-1, self.kernel_size[3] * 10 * 10)
            # x = self.fc1(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x))
        if (self.conv_num == 4 and self.batch_norm == True):
            # x = self.pool(F.relu(self.bn1(self.conv1(x))))
            # x = self.pool(F.relu(self.bn2(self.conv2(x))))
            # x = self.pool(F.relu(self.bn3(self.conv3(x))))
            # x = self.pool(F.relu(self.bn4(self.conv4(x))))
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))

            #x = self.pool(F.relu(self.conv5(x)))

            x = x.view(-1, self.kernel_size[4] * 8 * 8)

            x = self.fc1(x)
            # x = F.relu(self.fc2_bn(self.fc2(x)))
            # x = self.fc3(x)
        return x