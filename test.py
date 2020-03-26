import os
import glob


curr_path = os.getcwd()
data_path = os.path.join(curr_path, 'data')
train_path = os.path.join(data_path, 'traindata')
test_path = os.path.join(data_path, 'testdata')
label_path = os.path.join(data_path, 'Label2Names.csv')

data = {}

data['train'] = []
data['val'] = []
train_label_path = sorted(glob.glob(os.path.join(train_path, '*', '*')))

for t in train_label_path:
    print(t)

