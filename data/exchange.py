import os,shutil
rootpath = "C:/Users/cheng/Desktop/python/defect_yolov8/ultralytics-main/data/"
imgtrain = rootpath + "images/train/"
imgval = rootpath + "images/val/"
labeltrain = rootpath + "labels/train/"
labelval = rootpath + "labels/val/"
if not os.path.exists(imgtrain):
    os.makedirs(imgtrain)
if not os.path.exists(imgval):
    os.makedirs(imgval)
if not os.path.exists(labeltrain):
    os.makedirs(labeltrain)
if not os.path.exists(labelval):
    os.makedirs(labelval)

f = open(rootpath + "dataSet/train.txt", "r")
lines = f.readlines()
for i in lines:
    shutil.move(rootpath + "images/train/" + str(i).replace('\n','') + ".jpg", imgtrain + str(i).replace('\n','') + ".jpg")
    shutil.move(rootpath + "labels/train/" + str(i).replace('\n', '') + ".txt", labeltrain + str(i).replace('\n', '') + ".txt")

f = open(rootpath + "dataSet/val.txt","r")
lines = f.readlines()
for i in lines:
    shutil.move(rootpath + "images/val/" + str(i).replace('\n','') + ".jpg", imgval + str(i).replace('\n', '') + ".jpg")
    shutil.move(rootpath + "labels/val/" + str(i).replace('\n','') + ".txt", labelval + str(i).replace('\n', '') + ".txt")
shutil.move(rootpath + "dataSet/train.txt", rootpath + "train.txt")
shutil.move(rootpath + "dataSet/trainval.txt", rootpath + "trainval.txt")
shutil.move(rootpath + "dataSet/test.txt", rootpath + "test.txt")
shutil.move(rootpath + "dataSet/val.txt", rootpath + "val.txt")