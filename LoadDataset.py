from angle_data import AngleSteering

dataset = AngleSteering(csv_file="",root_dir="",transform=)

train_set,test_set = torch.utils.data.random_split(dataset,[,])

train_loader = Dataloader(dataset = train_set,batch_size = batch_size, shuffle = True)
test_loader = Dataloader(dataset = test_set,batch_size = batch_size, shuffle = True)