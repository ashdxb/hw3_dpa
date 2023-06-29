import torch
import torch.nn.functional as F
from dataset import HW3Dataset
from gat_model import GAT
from tqdm import tqdm
import os

# one training step
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].squeeze())
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate the model on the validation set
def val(model):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = float(pred[data.val_mask].eq(data.y[data.val_mask].squeeze()).sum().item())
    acc = correct / data.val_mask.shape[0]
    return acc

# Initialize the model and load the best model
def init_vars():
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    if os.path.exists(accs_losses_save_path):
        accs, losses = torch.load(accs_losses_save_path)
    else:
        accs = []
        losses = []

    best_acc = val(model)
    return accs, losses, best_acc

# setting up variables
model_save_path = 'best_model_gat.pt'
accs_losses_save_path = 'accs_losses_gat.pt'
accs_save_path = 'accs_gat.png'
losses_save_path = 'losses_gat.png'

torch.cuda.empty_cache()

# load the data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = HW3Dataset(root='data/hw3/')
data = dataset[0].to(device)

# initialize the model and optimizer
model = GAT(data.num_features, 48, dataset.num_classes).to(device) # Model that got 0.6256, should be 48
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Model that got 0.6256, should be 0.001

# initialize the variables
accs, losses, best_acc = init_vars()

# train the model
epochs = 470
def trainer(epochs, best_acc=best_acc, accs=accs, losses=losses):
    for epoch in tqdm(range(epochs)):
        loss = train()
        losses.append(loss)
        acc = val(model)
        accs.append(acc)
        # save the best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_save_path)
            torch.save([accs, losses], accs_losses_save_path)
    return accs, losses, best_acc

accs, losses, best_acc = trainer(epochs)

def print_and_graph(accs, losses, best_acc):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(accs) + 1))

    # plot the accuracy
    plt.clf()
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, accs)
    plt.savefig(accs_save_path)

    # plot the loss
    plt.clf()
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, losses)
    plt.savefig(losses_save_path)

    # load the best model
    model.load_state_dict(torch.load(model_save_path))
    print('Best Accuracy: {:.4f}'.format(best_acc))
    print('Best Epoch: {}'.format(accs.index(best_acc) + 1))
    print('Accuracy: {:.4f}'.format(best_acc))
    print('Epochs Trained: {}'.format(len(accs)))

print_and_graph(accs, losses, best_acc)