import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim

from dataloader import MnistBags
from model import Attention, Attention_instance_loss

parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--model', type=str, help='Choose b/w A (attention) and B (attention with instance loss)')
args = parser.parse_args()

epochs=20
lr=5e-4
reg=10e-5
target_number=9
mean_bag_length=10
var_bag_length=2
num_bags_train=200
num_bags_test=500

train_loader = data_utils.DataLoader(MnistBags(target_number=target_number,
                                               mean_bag_length=mean_bag_length,
                                               var_bag_length=var_bag_length,
                                               num_bag=num_bags_train,
                                               train=True),
                                     batch_size=1,
                                     shuffle=True)

test_loader = data_utils.DataLoader(MnistBags(target_number=target_number,
                                              mean_bag_length=mean_bag_length,
                                              var_bag_length=var_bag_length,
                                              num_bag=num_bags_test,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False)

if args.model == 'A':
    model = Attention().cuda()
    print('choosing the model from "Attention based deep multiple instance learning"')
elif args.model == 'B':
    model = Attention_instance_loss().cuda()
    print('choosing my model with instance loss')
    
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=reg)
criterion = nn.BCEWithLogitsLoss(reduction='none')

def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]

        data, bag_label = data.cuda(), bag_label.float().cuda()

        # reset gradients
        optimizer.zero_grad()
        
        output, loss=model(data, criterion,bag_label)
        train_loss += loss.item()
        error = (output>0)!=bag_label
        train_error += error.item()

        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    # print('Epoch: {}, Loss: {:.4f}, Train error: {:.2f}%'.format(epoch, train_loss, 100*train_error))


def test(rep):
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        data, bag_label = data.cuda(), bag_label.float().cuda()
        
        output, loss=model(data, criterion,bag_label)
        test_loss += loss.item()
        error = (output>0)!=bag_label
        test_error += error.item()


    test_error /= len(test_loader)
    test_loss /= len(test_loader)
    
    # torch.save(model.state_dict(),'model.pt')

    print('rep {}, Loss: {:.4f}, Test error: {:.2f}%'.format(rep, test_loss, 100*test_error))
    return test_loss, 100*test_error


if __name__ == "__main__":
    print('repeating 10 times...')
    losses=[]
    errors=[]
    for i in range(1,11):
        for epoch in range(1, epochs + 1):
            train(epoch)
        test_loss,test_error=test(i)
        losses.append(test_loss)
        errors.append(test_error)
    loss_std,loss_mean=torch.std_mean(torch.tensor(losses), unbiased=True)
    error_std,error_mean=torch.std_mean(torch.tensor(errors), unbiased=True)
    avg_error=torch.tensor(errors).mean()
    print(f'Final result, loss: {loss_mean.item():.2f} ± {loss_std.item():.2f}, error: {error_mean.item():.2f}% ± {error_std.item():.2f}%')
    print()
