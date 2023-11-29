import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')


















 # your work is start her

# ### load our data 
# import numpy as np
# import random
# import json

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from nltk_utils import bag_of_words, tokenize, stem
# from model import NeuralNet

# with open("intents.json","r") as f:
#     intents=json.load(f)

# ## check for the intents
# # print(intents)

# all_words=[]
# tags=[]
# xy=[]

# for intent in intents['intents']:
#     tag=intent['tag']
#     tags.append(tag)
#     for pattern in intent ["patterns"]:  ## toknize our pattern
#          w=tokenize(pattern)             ##arry 
#          all_words.extend(w)              ## use extend caus its arry
#          xy.append((w,tag))                 ## put the toknize patter


# ignor_words=["?","!","@",".",",","$","#",":"]
# all_words=[stem(w)for w in all_words if w not in ignor_words ] ## if w not in ignor_words ###this is list comprehension tool  #### to excloude yhe ignor words
# ## print(all_words)  ###to check the ignor words

# all_words=sorted(set(all_words)) ##  trik to rimove the dublicate element
# tags=sorted(set(tags))##not naccecry
# #print(tags) ## to print the diffrent tags for cheak

# ### creat the bag _words

# x_train=[] ##the tax or the associated no for each tags
# y_train=[]
# for ( pattern_sentece,tag) in xy:
#     bag=bag_of_words(pattern_sentece,all_words)
#     x_train.append(bag)

#     label=tags.index(tag)
#     y_train.append(label) ##crossEntropyloss

# x_train=np.array(x_train)
# y_train=np.array(y_train)

# class chatDataset(Dataset):
#     def __init__(self) :
#         self.n_samples=len(x_train)
#         self.x_data=x_train
#         self.y_data=y_train
        
#         #dataset[idx]
#         def __getitem__(self,index):
#             return self.x_data[index],self.y_data[index]
        
#         def __len__(self):
#             return self.n_samples
# #hyper parameters 
# num_epochs = 1000
# batch_size = 8
# learning_rate = 0.001
# input_size = len(x_train[0]) ## lenrgth of each bag of words
# hidden_size = 8
# output_size = len(tags)
# print(input_size, len(all_words))
# print(output_size,tags)


# dataset=chatDataset()
# train_loader=DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NeuralNet(input_size, hidden_size, output_size) ##.to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # Train the model
# for epoch in range(num_epochs):
#     for (words, labels) in train_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)
        
#         # Forward pass
#         outputs = model(words)
#         # if y would be one-hot, we must apply
#         # labels = torch.max(labels, 1)[1]
#         loss = criterion(outputs, labels)
        
#         # Backward pass and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     if (epoch+1) % 100 == 0:
#         print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# print(f'final loss: {loss.item():.4f}')

# data = {
# "model_state": model.state_dict(),
# "input_size": input_size,
# "hidden_size": hidden_size,
# "output_size": output_size,
# "all_words": all_words,
# "tags": tags
# }

# FILE = "data.pth"
# torch.save(data, FILE)

# print(f'training complete. file saved to {FILE}')
        
        
        

        



