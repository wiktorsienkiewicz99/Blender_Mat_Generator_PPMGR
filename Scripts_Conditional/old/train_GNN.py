import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Load the JSON data
json_file_path = 'C:/Users/hyperbook/Desktop/PPMGR/Projects/Results/database/extracted_materials.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Custom Dataset Class
class BlenderMaterialDataset(Dataset):
    def __init__(self, data):
        self.data = self._preprocess_data(data)
        self.node_type_to_idx = self._create_node_type_to_idx()
    
    def _preprocess_data(self, data):
        # Convert the data into a list of dictionaries with 'Nodes' and 'Edges' keys
        preprocessed_data = []
        for material_name, material in data.items():
            preprocessed_data.append({
                'Nodes': material['Nodes'],
                'Edges': material['Edges']
            })
        return preprocessed_data
    
    def _create_node_type_to_idx(self):
        # Create a mapping from node types to unique indices
        node_types = set()
        for material in self.data:
            for node in material['Nodes']:
                node_types.add(node['Type'])
        return {node_type: idx for idx, node_type in enumerate(node_types)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        material = self.data[idx]
        node_types = [self.node_type_to_idx[node['Type']] for node in material['Nodes']]
        edges = material['Edges']
        
        edge_index = []
        for edge in edges:
            src = edge['Source'].split('.')[0]
            tgt = edge['Target'].split('.')[0]
            src_idx = next(i for i, node in enumerate(material['Nodes']) if node['Name'] == src)
            tgt_idx = next(i for i, node in enumerate(material['Nodes']) if node['Name'] == tgt)
            edge_index.append([src_idx, tgt_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_types = torch.tensor(node_types, dtype=torch.float).unsqueeze(1)
        
        return Data(x=node_types, edge_index=edge_index)

# Create Dataset and DataLoader
dataset = BlenderMaterialDataset(data)

# Custom collate function
def collate_fn(batch):
    return Batch.from_data_list(batch)

train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

input_dim = 1
hidden_dim = 64
output_dim = 37

model = GNN(input_dim, hidden_dim, output_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        target = data.x.squeeze().long()  # Ensure target is of shape [batch_size] and type long
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
