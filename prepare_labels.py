# coding: utf-8
import pandas as pd
import pickle as pkl

path = 'C:/Users/jimch/Documents/NYU/data/'


def convert_depth(x):
    if x == '>':
        return 1
    elif x == '<':
        return -1
    else:
        return 0


def convert_labels(path):
    raw_labels = pd.read_csv(path, header=None, skiprows=1)
    names = raw_labels[raw_labels[0].apply(lambda x: x[0] == '.')]
    raw_labels[4] = raw_labels[4].apply(convert_depth)
    labels = []
    for i in range(len(names.index)):
        if i+1 >= len(names.index):
            points = raw_labels[names.index[i]+1:].values.astype('float32')
        else:
            points = raw_labels[names.index[i]+1:names.index[i+1]].values.astype('float32')
        label = {}
        label['name'] = names.iloc[i][0].split('/')[-1]
        label['x_A'] = points[:, 0]
        label['y_A'] = points[:, 1]
        label['x_B'] = points[:, 2]
        label['y_B'] = points[:, 3]
        label['ordinal_relation'] = points[:, 4]
        # print(len(points))
        # if len(points) != 800:
        #     continue
        labels.append(label)
    return labels


def save(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)
       

def load(path):
    with open(path, 'rb') as f:
        return pkl.load(f)  


labels = convert_labels(path+'train_labels/750_train_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv')
print(len(labels))
print(len(labels[0]))
print(len(labels[0]["ordinal_relation"]))

labels_val = convert_labels(path+'val_labels/45_validate_from_795_NYU_MITpaper_train_imgs_800_points_resize_240_320.csv')

print(len(labels_val))
print(len(labels_val[0]))
print(len(labels_val[0]["ordinal_relation"]))

labels_test = convert_labels(path+'test_labels/654_NYU_MITpaper_test_imgs_orig_size_points.csv')

print(len(labels_test))
print(len(labels_test[0]))
print(len(labels_test[0]["ordinal_relation"]))

save(labels, path+'labels_train.pkl')
save(labels_val, path+'labels_val.pkl')
save(labels_test, path+'labels_test.pkl')

labels = load(path+'labels_train.pkl')
labels_val = load(path+'labels_val.pkl')
labels_test = load(path+'labels_test.pkl')
