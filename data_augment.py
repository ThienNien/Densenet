import torch 
import torchvision.transforms as transforms
from torch.utils import save_image
from LoadDataset import AngleSteering

my_transforms =  transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees = 45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p =0.05),
    transforms.RandomGrayscale(p =0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean =[0.0,0.0,0.0],std = [1.0,1.0,1.0]) #(value -mean)/std
])

dataset = AngleSteering(csv_file = "",root_dir = "",transform = my_transforms)

img_num = 0
for _ in range(10):
    for img,label in dataset:
        save_image(img,'img'+str(img_num)+"Augment.png")
        img_num += 1
        