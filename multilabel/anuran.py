# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import fbeta_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib qt

#preprocessing
train = pd.read_csv('Frogs_MFCCs.csv')
train = train.iloc[np.random.permutation(len(train))].reset_index(drop=True)   
#train.isnull().values.any()
train_y = train[['Family','Genus','Species']]
train.drop(['Family','Genus','Species'],axis=1, inplace =True)
train_y = pd.get_dummies(train_y)

#normalisation
train = (train-train.mean())/train.std()

#making test set
test = train.drop(train.index[:6500])
train.drop(train.index[6500:], inplace=True)
test_y = train_y.drop(train_y.index[:6500])
train_y.drop(train_y.index[6500:], inplace=True)

#preparing dataset 
train = torch.FloatTensor(train.values)
train_y = torch.FloatTensor(train_y.values)
test = torch.FloatTensor(test.values)
test_y = torch.FloatTensor(test_y.values)
torch_dataset = Data.TensorDataset(train, train_y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=64, shuffle=True)

#designing architecture
class multi_nn(torch.nn.Module):
    def __init__(self, n_hidden1, n_hidden2, n_output):
        super(multi_nn, self).__init__()
        self.hidden1 = torch.nn.Linear(23, n_hidden1)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.output = torch.nn.Linear(n_hidden2, n_output)
    
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        out = self.output(x)
        return out
    
#initialising network, loss, optimiser
net = multi_nn(11,11,22)
loss_func = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#training loop
tr_losses = []
ts_losses = []
f2_scores = []
for epoch in range(540):
    for b_x, b_y in loader:
        out = net(b_x)
        loss = loss_func(out, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_losses.append(loss)
    
    #prediction
    if (epoch %10 ==0):
        pred_y = net(test)
        ts_losses.append(loss_func(pred_y, test_y))
        idxs = np.argsort(pred_y.data.numpy(), axis=1)[:,-3]                      #calculate the indices for third highest value
        thd_max = [pred_y[i,idx] for i, idx in enumerate(idxs)]                   
        pred_y = pred_y.data.numpy()>=np.array(thd_max)[:,None]
        f2_scores.append(fbeta_score(test_y, pred_y, beta=2, average='samples'))
        comp_res = np.all(pred_y == test_y.data.numpy(), axis=1)
        n_wrng_pred = 15290 - sum(np.sum(pred_y==test_y.data.numpy(), axis=1))    #total no of wrong predictions
        wrng_pred = len(pred_y)-np.sum(comp_res)                                  #no of wrong predicted rows
        print('Epoch: ', epoch, '| Tr_loss: %.4f' % tr_losses[-1].data.numpy(),
              '| Ts_loss: %.4f' % ts_losses[-1].data.numpy(),
              '| Ts_F2Score: %.4f' % f2_scores[-1], ' |Wrong_Pred:', wrng_pred)

fig = plt.figure()
plt.subplot(2,2,1)
plt.plot(tr_losses, 'b')
plt.xlabel('Steps')
plt.ylabel('Train Loss')

plt.subplot(2,2,2)
plt.plot(f2_scores, 'r')
plt.xlabel('Epochs')
plt.ylabel('F2_Score')

plt.subplot(2,2,3)
plt.plot(ts_losses, 'g')
plt.xlabel('Epochs')
plt.ylabel('Val Loss')
plt.show()
