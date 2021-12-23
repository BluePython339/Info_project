import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import nn
import re
from collections import Counter

with open("quad_profile.bin", 'r') as prof:
    a = prof.readlines()
    base = {}
    for i in range(300):
        base[a[i].split(" ")[0]] = 0
    base["length"] = 0

def preprocess_text(text):
    n = 4
    text = text.lower()
    stripped = re.sub(r'([^A-Za-z ])+', "", text)
    prepped = re.sub(r'([^A-Za-z])+', "_", stripped)
    res = [prepped[i:i + n] for i in range(len(prepped) - n + 1)]
    length = len(prepped)
    counts = list(zip(Counter(res).keys(),Counter(res).values()))
    counts.sort(key=lambda tup: tup[1], reverse=True)
    it_count = base.copy()
    for i in counts:
        if i[0] in it_count.keys():
            it_count[i[0]] = i[1]

    it_count["length"] = length
    np.array([x[1] for x in it_count.items()])
    return torch.FloatTensor([x[1] for x in it_count.items()])


def load_in_dataset(fname):
    x = []
    y = []
    with open(fname, "r") as rdata:
        for i in rdata.readlines():
            a = list(map(int, i.split(",")))
            x.append(a[:-1])
            y.append(a[-1])
    return np.array(x), np.array(y)


class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,5)
        self.fc3 = nn.Linear(5,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


if __name__ == "__main__":
    x,y = load_in_dataset("dataset_train_quad_2.csv")
    trainset = dataset(x,y)
    valset = dataset(*load_in_dataset("dataset_val_quad_2.csv"))
    valloader = DataLoader(valset, batch_size=64, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    lr = 0.0001
    epochs = 300
    min_valid_loss = np.inf
    print(x.shape)
    model = Net(input_shape=x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    losses = []
    accur = []
    for i in range(epochs):
        losses = []
        accur = []
        for j, (x_train, y_train) in enumerate(trainloader):
            # calculate output
            output = model(x_train)

            # calculate loss
            loss = loss_fn(output, y_train.reshape(-1, 1))

            # accuracy
            predicted = model(torch.tensor(x, dtype=torch.float32))
            acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accur.append(acc.item())

        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in valloader:
            #if torch.cuda.is_available():
            #    data, labels = data.cuda(), labels.cuda()

            target = model(data)
            loss = loss_fn(target, labels.reshape(-1, 1))
            valid_loss = loss.item() * data.size(0)
        print("epoch {}\tloss : {}\t accuracy : {}\t validation loss {}".format(i, np.mean(losses), np.mean(accur), valid_loss/ len(valloader)))
        #print(f'Epoch {i} \t\t Training Loss: {np.mean(losses)} \t\t Validation Loss: {valid_loss / len(valloader)}')
        if min_valid_loss > valid_loss and min_valid_loss - valid_loss > 0.01:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model_2.pth')
            #exit()

        #print("epoch {}\tloss : {}\t accuracy : {}".format(i, np.mean(losses), np.mean(accur)))




