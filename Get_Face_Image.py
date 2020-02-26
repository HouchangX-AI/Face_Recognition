# 导入包
import os 
import cv2 
import time 
import dlib 
from Read_Data_Image import readAllImg

#从源路径中读取所有图片放入一个list，然后逐一进行检查，把其中的脸扣下来，存储到目标路径中
def readPicSaveFace(sourcePath,objectPath,min_p,max_p,size,number,*suffix):
    '''
    输入：
    1，sourcePath 输入路径（姓名文件夹）
    2，objectPath 输出路径
    3，min_p 最小人脸数量
    4，max_p 最大人脸数量
    5，size 人脸输出大小
    6，number 是第几个人的脸
    7，*suffix 图片格式
    '''
    detector = dlib.get_frontal_face_detector()

    #读取照片,注意第一个元素是文件名
    resultArray=readAllImg(sourcePath,*suffix)

    #对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
    count = 0

    # 遍历图片，保存图片中的人脸
    face_list = list()
    for i in resultArray:
        # 若人脸数达到上限，则停止生成人脸图
        if count < max_p:
            if type(i) != str:
                # 检测人脸
                dets = detector(i, 1)
                # 判断人脸数量（若有多个人脸，则说明图片中不止一个人，与人脸识别的单人数据相悖，所以排除）
                if len(dets) == 1:
                    face = dets[0]
                    left = face.left()
                    top = face.top()
                    right = face.right()
                    bottom = face.bottom()
                    
                    # resize图片到指定大小，并保存图片(因为dlib可能预测出负值，所以加了个try)
                    try:
                        f = cv2.resize(i[top:bottom, left:right], (size, size))
                        face_list.append(f)
                        count += 1
                    except:
                        pass
        else:
            break
    # 如果这个文件夹中的可用人脸数足够多
    if len(face_list) > min_p:
        # 建立输出路径，创建文件夹
        file_dir = sourcePath.split('/')
        one_name = file_dir[-1]
        save_dir = os.path.join(objectPath, one_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        # 把每一个人脸保存起来
        for num,p in enumerate(face_list):
            # 用时间戳和图片数来命名
            listStr = [str(int(time.time())), str(num)]
            fileName = ''.join(listStr)
            print('第',number+1,'人，第',num,'张图：',save_dir+'/'+fileName+'.jpg')
            # 保存图片
            cv2.imwrite(save_dir+'/'+fileName+'.jpg', p)
        print('Already read '+str(len(face_list))+' Faces to Destination '+objectPath)
        return(str(len(face_list))+'_img')
    else:
        return('no_img')

def main(in_dir,out_dir,min_p,max_p,num,size):

    data_dir = in_dir
    humans_list = os.listdir(data_dir)
    number = 0
    for one_human in humans_list:
        if one_human != '.DS_Store' and number < num:
            one_human_dir = os.path.join(data_dir, one_human)
            result = readPicSaveFace(one_human_dir, out_dir, min_p, max_p, size, number,'.jpg', '.JPG', 'png','PNG')
            if result != 'no_img':
                number += 1

if __name__ == '__main__':
    in_dir = 'Files/test'
    out_dir = 'Files/data_50man_300p_224'
    min_p = 300
    max_p = 301
    size = 224
    num = 50
    main(in_dir,out_dir,min_p,max_p,num,size)
