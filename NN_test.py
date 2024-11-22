import torch
import torch.nn as nn
import random

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 350)
        self.fc2 = nn.Linear(350, 350)
        self.fc3 = nn.Linear(350, 128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def context(self, x):
        x = self.relu(self.fc1(x))
        print("After fc1 and relu:", x)
        if self.training:
            x = self.dropout(x)
            print("After dropout:", x)
        x = self.relu(self.fc2(x))
        print("After fc2 and relu:", x)
        x = self.fc3(x)
        print("After fc3:", x)
        x = self.relu(x)
        print("After fc3 and relu:", x)
        return x

model = DNN(input_dim=40)
model_path = "dnn_model.pth"  
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()
test = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.3, 5083.954]
#current context: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.3, 5083.954]
#current context: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 8.1, 3827.658]
def extract_context(dnn_model, X_tensor):
    with torch.no_grad():
        features = dnn_model.context(X_tensor).numpy()
    return features

X_tensor = torch.tensor(test, dtype=torch.float32)
x_ta = extract_context(model, X_tensor.unsqueeze(0))
print(x_ta)



#top_image_urls = [(df.iloc[arm, -1], df.index[arm]) for arm in top_arms]