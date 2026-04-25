import matploylib.pyplot as plt
import torch 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataloader, TensorDataset

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self,x):
        return self.linear(x)
    
def load_data(test_size:float = 0.2, random_state:int = 4):
    dataset = fetch_california_housing()
    features = dataset.data
    targets = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(features, targets,test_size=0.2,random_state=random_state)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_tensor = 

    
    
