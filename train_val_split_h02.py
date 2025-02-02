import os
import numpy as np
import time
from sklearn.model_selection import train_test_split

import sklearn
import shutil
import yaml


def GetFileList(dirName1, fname2FullPathDic):
    cnt = 0
    for path, subdirs, files in os.walk(dirName1):

        for name in files:
            extension = os.path.splitext(name)[1]
            if extension == '.png':
                if name not in fname2FullPathDic:
                    value = os.path.join(path, name)
                    fname2FullPathDic[cnt] = value


def copyFromBaseline(dirName1, fname2FullPathDic):
    GetFileList(dirName1, fname2FullPathDic)
    for i in range(len(fname2FullPathDic)):
        fn = fname2FullPathDic[i]
        fnout1 = fn.replace(str1, str2)
        shutil.copy(fn, fnout1)


def data_split(examples, labels, train_frac, random_state=None):
    ''' https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    param data:       Data to be split
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'valid': (1-train_frac) / 2
        'test':  (1-train_frac) / 2
    Eg: passing train_frac=0.8 gives a 80% / 10% / 10% split
    '''

    assert train_frac >= 0 and train_frac <= 1, "Invalid training set fraction"

    X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(
        examples, labels, train_size=train_frac, random_state=random_state)

    return X_train, X_val


def copyTask(TrainGrovePath, coll, root):
    global i
    for filename in coll:
        i = i + 1
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ARN, extension = os.path.splitext(filename)
        file = os.path.join(TrainGrovePath, ARN)
        temp1 = file + "_" + str(i) + "_" + timestr + "_" + extension
        sourcePath = root + "/" + filename
        shutil.copy(sourcePath, temp1)


def splitingData(category, f_names, train_frac, baseDataPath, root):
    cnt = len(f_names)
    print('count of file=', cnt)
    if (cnt > 0):
        X_train, X_val = data_split(f_names, f_names, train_frac)
        # print('X_train=', X_train)

        for ob in ['train', 'val']:
            dataPath = os.path.join(baseDataPath, ob + "/" + category + "/")
            if (ob == 'train'):
                listOfFile = X_train
            if (ob == 'val'):
                listOfFile = X_val

            # print(dataPath)

            copyTask(dataPath, listOfFile, root)


def CreateDir(parent_dir, directory):
    path = os.path.join(parent_dir, directory)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)  # Removes all the subdirectories!
        os.makedirs(path)


def main():
    # conf_file = r'config/hierarchy_1_config.yaml'
    conf_file = r'hierarchy_1_config2.yaml'
    with open(conf_file) as file:
        ConfigFile = yaml.load(file, Loader=yaml.FullLoader)

    pathBase = ConfigFile['parameters']['BasePath']
    FolderNameFrom = ConfigFile['parameters']['AllData']
    CreatedFolder = ConfigFile['parameters']['SplitData']
    train_frac = ConfigFile['parameters']['train_frac']
    output_train_val_root = ConfigFile['parameters']['output_train_val_root']

    myRoot = pathBase + "/"+FolderNameFrom + "/"
    mr = output_train_val_root + "/" + CreatedFolder

    directory = "data"
    baseDataPath = mr + "/" + directory + "/"
    myCategory = ["train", "val"]

    print("Task Started ...")

    result = []

    for x in os.walk(myRoot):
        # print(x)
        result = x[1]
        break

    parent_dir = mr
    CreateDir(parent_dir, directory)
    for value in myCategory:
        directory = value
        parent_dir = mr + r"/data/"
        CreateDir(parent_dir, directory)
        for ob in result:
            subdir = parent_dir + value
            CreateDir(subdir, ob)

    path = myRoot
    print(path)
    for root, d_names, f_names in os.walk(path):
        # print(root, d_names, f_names)

        nstr = root.split("/")
        category = nstr[len(nstr) - 1]

        cnt = len(f_names)
        if cnt > 0:
            print('f_names=', f_names)
            value = np.random.permutation(f_names)  # convert from list to ndarray and randomize the order
            num = len(
                value)  ## num indicates the number of image to select from folder it can varies but less than equal to len(value)
            f_names = value[:num]

            # foo = ['.zip', 'Processed_Image']  # Search the presence of file not to be included in computaton
            # out = [v for i, v in enumerate(f_names) for ob in foo if ob in v]
            #
            # f_names = np.delete(f_names, np.argwhere(f_names == out))  ## remove the file name from the list

            splitingData(category, f_names, train_frac, baseDataPath, root)  # function to perform splitting task

    print("Task Completed ...")


i = 0
if __name__ == '__main__':
    main()
