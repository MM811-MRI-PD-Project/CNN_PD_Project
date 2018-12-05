# Read result.csv and plot
import matplotlib.pyplot as plt
import csv

filename = 'result_wm60_bn.csv'

result_file = open(filename,'r')
result = csv.reader(result_file,delimiter=',')
epoch = []
y_train = []
y_val=[]
y_train_loss = []
y_val_loss = []

for row in result:
    if float(row[0]) >=1:
        epoch.append(int(row[0]))
        y_train.append(float(row[1]))
        y_val.append(float(row[2]))
        y_train_loss.append(float(row[3]))
        y_val_loss.append(float(row[4]))
    else:
        print(row[0])

result_file.close()  

fig = plt.figure(1)
x = epoch
plt.plot(x, y_train)
plt.plot(x, y_val)
plt.legend(['train_acc', 'val_acc'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.savefig('Acc.png')

fig = plt.figure(2)
plt.plot(x, y_train_loss)
plt.plot(x, y_val_loss)
plt.ylim(0,2)
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.xlabel('Epon')
plt.ylabel('Loss')
plt.savefig('Loss,png')

plt.show()
