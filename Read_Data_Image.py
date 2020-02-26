import os 
import cv2 
import numpy as np 

#根据输入的文件夹绝对路径，将该文件夹下的所有指定suffix的文件读取存入一个list,该list的第一个元素是该文件夹的名字
def readAllImg(path,*suffix):
    try:
        # 将人名放入返回的矩阵
        s = os.listdir(path)
        resultArray = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)
        # 将每一张图放入矩阵
        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                resultArray.append(img)
    except IOError:
        print("Error")
    else:
        print('检查',path,'文件数量')
        return resultArray

#输入一个字符串一个标签，对这个字符串的后续和标签进行匹配
def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

#输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
#返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)

def read_file(path):
    img_list = []
    label_list = []
    name_list = []
    dir_counter = 0
    IMG_SIZE = 224

    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        if child_dir != '.DS_Store':
            child_path = os.path.join(path, child_dir)
            name_list.append(child_path)
            for dir_image in  os.listdir(child_path):
                if endwith(dir_image,'jpg'):
                    img = cv2.imread(os.path.join(child_path, dir_image))
                    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    # recolored_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
                    img_list.append(resized_img)
                    label_list.append(dir_counter)
            dir_counter += 1

    # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)

    return img_list,label_list,dir_counter,name_list

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list
