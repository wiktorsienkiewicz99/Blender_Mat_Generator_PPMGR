import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig

json_file_path = '../Projects/Results/database/extracted_materials.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MaterialDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.materials = list(data.items())
    
    def __len__(self):
        return len(self.materials)
    
    def __getitem__(self, idx):
        material_name, material = self.materials[idx]
        nodes = material['Nodes']
        
        input_text = " ".join([f"{node['Name']} ({node['Type']})" for node in nodes])
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        }

dataset = MaterialDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=37)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Traini
def train():
    model.train()
    for epoch in range(10): 
        for batch in dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = torch.randint(0, 37, (input_ids.size(0),))  # Dummy target labels for debugging

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits

            print(f"Epoch: {epoch+1}, Loss: {loss.item()}, Logits Shape: {logits.shape}, Labels Shape: {labels.shape}")
            
            # Backward pass
            loss.backward()
            optimizer.step()

train()

def generate_nodes(model, tokenizer, input_text, max_len=128):
    inputs = tokenizer(
        input_text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    model.eval() 
    with torch.no_grad():
        outputs = model(**inputs)
  
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    predicted_material_name = list(data.keys())[predicted_class]
    return data[predicted_material_name]

test_input_text = "Fresnel (FRESNEL) Mix Shader (MIX_SHADER) ColorRamp (VALTORGB)"
predicted_nodes = generate_nodes(model, tokenizer, test_input_text)
print(predicted_nodes)