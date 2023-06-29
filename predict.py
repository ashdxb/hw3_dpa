from dataset import HW3Dataset
from gat_model import GAT
import torch

# setting up variables
model_save_path = './best_model_gat.pt'

# load the data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = HW3Dataset(root='data/hw3/')
data = dataset[0].to(device)

# initialize the model and optimizer
model = GAT(data.num_features, 48, dataset.num_classes).to(device) # Model that got 0.6256, should be 48
model.load_state_dict(torch.load(model_save_path))

# predict
model.eval()
out = model(data.x, data.edge_index)
pred = out.argmax(dim=1)

# save the prediction
import pandas as pd
df = pd.DataFrame({'idx': list(range(data.x.shape[0])), 'prediction': pred.tolist()})
df.to_csv('prediction.csv', index=False)