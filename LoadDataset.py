from angle_data import AngleSteering
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
batch_size = 16

dataset = AngleSteering(csv_file="data_angle.csv",root_dir="data",transform=transforms.ToTensor())

train_set,test_set = torch.utils.data.random_split(dataset,[1028,400])

train_loader = DataLoader(dataset = train_set,batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set,batch_size = batch_size, shuffle = True)

for i,(data,label) in enumerate(train_loader):
    pass