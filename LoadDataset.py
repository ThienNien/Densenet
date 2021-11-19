from angle_data import AngleSteering
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
batch_size = 16
transform_t = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
csv_path =r"C:\Users\tncup\Desktop\Densenet\data_angle.csv"
data = r"C:\Users\tncup\Desktop\Densenet\data"
dataset = AngleSteering(csv_file=csv_path,root_dir=data,transform=transform_t)

train_set,test_set = torch.utils.data.random_split(dataset,[1028,400])

train_loader = DataLoader(dataset = train_set,batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set,batch_size = batch_size, shuffle = True)

classes = ( '-21', '-15', '-5','0', '5', '15', '21')

