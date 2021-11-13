from angle_data import AngleSteering
from torch.utils.data import DataLoader
import torch
batch_size = 16

dataset = AngleSteering(csv_file="C:\Users\tncup\Desktop\AI\car\Csv\data_angle",root_dir="C:\Users\tncup\Desktop\AI\car\Csv\data",transform=)

train_set,test_set = torch.utils.data.random_split(dataset,[,])

train_loader = DataLoader(dataset = train_set,batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set,batch_size = batch_size, shuffle = True)