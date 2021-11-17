from angle_data import AngleSteering
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
batch_size = 16
transform_re = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
csv_path =r"C:\Users\tncup\Desktop\Densenet\data_angle.csv"
data = r"C:\Users\tncup\Desktop\Densenet\data"
dataset = AngleSteering(csv_file=csv_path,root_dir=data,transform=transform_re)

train_set,test_set = torch.utils.data.random_split(dataset,[1028,400])

train_loader = DataLoader(dataset = train_set,batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set,batch_size = batch_size, shuffle = True)

print(train_set,test_loader)