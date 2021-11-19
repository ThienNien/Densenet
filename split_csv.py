import csv
import os
list_angle = []
with open('data_angle.csv','r') as AngleList:
    csv_angle = csv.reader(AngleList)
    for row in csv_angle:
        list_angle.append(row)

#print(list_angle[0:10])
train_list =[]
test_list =[]
for idx,item in enumerate(list_angle):
    if idx%4 == 0:
        test_list.append(item)
    else:
        train_list.append(item)

print(len(test_list),"is length of test set")
print(len(train_list),"is length of train set")
print(test_list[4:6])
with open('train_set.csv','w',newline='') as train:
    file_data = csv.writer(train)
    for item in train_list:
        file_data.writerow(item)
        
with open('test_set.csv','w',newline='') as test:
    file_data = csv.writer(test)
    for item in test_list:
        file_data.writerow(item)
