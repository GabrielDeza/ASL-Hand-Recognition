from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from torchsummary import summary
from model import Net

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K']
folder = 'asl_images'
data_set_size = 1370
own_data = False
if folder == 'asl_gd':
    own_data = True

"""------------------------------Hyper-parameters initialization------------------------------"""
learning_rate = 0.1
batch_size = 32
epoch_num = 100
torch.manual_seed(1)
eval_every = 30

"""------------------------------Helper Functions------------------------------"""
def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def un_encode(one_hot,size):
    one_hot_list = one_hot.tolist()
    corr_labels = []
    for i in range(size):
        corr_labels.append(one_hot_list[i].index(max(one_hot_list[i])))
    return corr_labels

def label_one_hot(x):
    enc = torch.zeros(10)
    enc[x] = 1.0
    return enc

def evaluate(loader, net,loss_fnc):
    correct = 0
    total = 0
    cnt =0
    running_loss =0
    with torch.no_grad():
        for i,data in enumerate(loader,0):
            cnt += 1
            images, labels = data
            outputs = net(images)
            loss = loss_fnc(input = outputs, target = labels)
            running_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += (predicted == labels.max(axis=1)[1]).sum().item()
            correct += (predicted == labels).sum().item()
    return correct/len(loader.dataset), running_loss/cnt

def get_data(dirname):
    data_set_size = 1370
    if own_data == True:
        data_set_size = 30
        validation = []
    norm_dataset = datasets.ImageFolder(f"./{folder}", transform = transforms.ToTensor())
    loader = DataLoader(norm_dataset, batch_size =data_set_size, shuffle = False)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize( mean = (0,0,0),std = (1,1,1))])
    #target_transform = transforms.Compose([transforms.Lambda(lambda x: label_one_hot(x))])
    dataset = datasets.ImageFolder(f"./{folder}", transform = transform)#, target_transform = target_transform)
    if own_data == False:
        training, validation = train_test_split(dataset, test_size = 0.2, random_state =40)
    if own_data == True:
        training = dataset
    return training, validation

"""------------------------------Loading the Input Data------------------------------"""

train,valid = get_data(folder)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
if own_data == False:
    valid_data = DataLoader(valid, batch_size = batch_size, shuffle = True)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K']

"""------------------------------CNN Initialization------------------------------"""
conv_net = Net(32, [10,10,10,10,10], 4,True)
summary(conv_net, input_size=(3, 56, 56))
"""------------------------------Training Loop------------------------------"""


#loss_func = nn.MSELoss(reduction='sum')
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(conv_net.parameters(), lr = learning_rate)

start = time.time()
training_loss = []
ACC_valid = []
ACC_train = []
valid_loss_list =[]
true = []
predict =[]
for epoch in range(epoch_num):
    q = 0
    running_loss = 0.0
    running_loss_valid = 0.0
    running_acc = 0.0
    valid_running_acc = 0.0
    final_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = conv_net(inputs)

        loss = loss_func(input = outputs, target = labels)
        # print("Loss:", loss)
        running_loss += loss

        loss.backward()
        optimizer.step()
        # print statistics
        if i % eval_every == (eval_every-1):
            correct = 0
            #labels_unencode = un_encode(labels, labels.shape[0])
            #labels_unencode = torch.FloatTensor(labels_unencode)
            #true.extend(labels_unencode.numpy())

            #training accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            predict.extend(predicted.numpy())
            total = labels.size(0)
            correct += (predicted == labels).sum().item()
            #correct = (predicted == labels.max(axis=1)[1]).sum().item()
            running_acc += (correct / total)
            # training loss calculation
            final_loss += (running_loss / eval_every)
            #validation accuracy calculation
            if own_data == True:
                print('[%d, %5d] loss: %.3f training accuracy: %.3f ' % (epoch + 1, i + 1, running_loss / eval_every, correct / len(inputs)))
            if own_data == False:
                valid_acc, valid_loss = evaluate(valid_data, conv_net,loss_func)
                running_loss_valid += (valid_loss)
                valid_running_acc += (valid_acc)
                print('[%d, %5d] loss: %.3f training accuracy: %.3f validation accuracy: %.3f' %(epoch + 1, i + 1, running_loss / eval_every, correct/len(inputs), valid_acc))
            running_loss = 0.0
            q = q + 1
    ACC_train.append(running_acc / q) #training accuracy list GOOD
    training_loss.append(final_loss / q)  # loss list
    if own_data == False:
        ACC_valid.append(valid_running_acc / q) #validation accuracy list GOOD
        valid_loss_list.append(running_loss_valid /q)

end = time.time()
print('Finished Training, Time Elapsed: %.2f' %(end-start))
epoch_number = [i for i in range(1,epoch_num+1)]
plt.subplot(1, 2, 1)
plt.title('Loss versus Epoch for everyones data')
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(epoch_number, training_loss)
plt.plot(epoch_number, valid_loss_list)
#plt.ylim([0,10])
plt.subplot(1, 2, 2)
plt.title('Accuracy versus Epoch for everyones data')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim([0,1.1])
plt.plot(epoch_number, ACC_train)
if own_data == False:
    plt.plot(epoch_number, ACC_valid)
plt.show()
if own_data == True:
    for i in range(0,len(ACC_train)-1):
        if (ACC_train[i] == 1):
            print("it reached max accuracy at epoch %d" %i)
if own_data == False:
    print(max(ACC_valid))
    print(ACC_valid.index(max(ACC_valid)))
print(confusion_matrix(true, predict))
torch.save(conv_net.state_dict(),'MyBest.pt')
