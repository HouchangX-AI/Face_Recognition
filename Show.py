'''
本文件为人脸识别效果展示文件。
'''

# 导入包
import cv2 # 图像处理
import csv # CSV文件读取
import dlib # 人脸检测
import numpy as np # 科学计算
from PIL import Image, ImageDraw, ImageFont # 在图片删写字
from keras.models import load_model # 导入模型

from Config import config

# 函数部分
def get_name_list(path):
    '''
    功能：从存储姓名列表的csv文件中获取姓名列表。
    输入：姓名列表csv文件路径。
    输出：姓名列表。
    '''
    name_list = list()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for one in row:
                name_list.append(one.split('/')[-1])
    return name_list

def get_model_and_perdict(image, model_path):
    '''
    功能：对图片中的人脸进行识别
    输入：dlib判定为人脸的图片
    输出：模型预测结果、预测结果人在姓名列表中的index
    '''
    # 导入模型（注意：若需快速重复调用人脸识别函数，则应把导入模型放在循环之外，只导入一次即可）
    model = load_model(model_path)
    # 图片预处理
    P_image = processing_pictures(image)
    # 模型预测
    pred = model.predict(P_image)
    # 找出预测结果最大值所在的index
    max_index = np.argmax(pred)
    return pred[0], max_index

def processing_pictures(image):
    '''
    功能：图片处理（resize、reshape、astype、归一化）
    输入：处理前的图片
    输出：处理后的图片
    '''
    img = cv2.resize(image,(config['image_size'], config['image_size']))
    img = img.reshape((1, config['image_size'], config['image_size'], 3))
    img = img.astype('float32')
    img = img/255.0
    return img

def detect_face(image, detector):
    '''
    功能：使用dlib进行人脸检测，使用训练的模型进行人脸识别
    输入：图片、dlib检测器
    输出：人脸框做表列表、人名列表
    '''
    # 使用dlib人脸检测器检测图片中的人脸
    dets = detector(image, 1)
    # 预留人脸框、人名列表
    face_box = list()
    face_name = list()
    # 读取姓名列表
    name_list = get_name_list(config['name_list_path'])
    
    # 对检测到的人脸进行预测
    if len(dets)>0:
        for face in dets:
            # 获取人脸框点位信息，并传入人脸框列表
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            face_box.append([top,bottom,left,right])
            
            # 把人脸部分从大图中切下来
            img = image[top:bottom, left:right]
            # 使用模型进行预测
            pred, max_index = get_model_and_perdict(img, config['model_file_path'])
            # 预测置信度大于阈值（0.7）则判断为是这个人，否则算陌生人
            if pred[max_index] > config['confidence_level']: 
                face_name.append(name_list[max_index])
            else:
                face_name.append('Stranger')
    return face_box, face_name

def add_txt(image,size,w,h,txt):
    '''
    功能：向图片上写字。
    输入：图片、字体大小、字体左上角x坐标、字体左上角y坐标、要写的字
    输出：写好字的图片
    注意：这里image的输入和输出都是PIL格式的，不是Opencv，因为OpenCV写中文会乱码。。
    '''
    # 定义字体字号
    setFont = ImageFont.truetype(config['font_type'], size)
    # 定义画板
    draw = ImageDraw.Draw(image)
    # 写字
    draw.text((w, h), txt, font=setFont, fill=config['font_color'])
    return image

def main(img_path,show=True,save_img=False,save_path='image.jpg'):
    '''
    功能：主函数，串联人脸检测-识别-展示全流程
    输入：原始图片路径、是否展示图片、是否保存图片、保存图片地址
    输出：无
    '''
    img = cv2.imread(img_path)
    # 定义dlib人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 进行人脸识别、人脸检测
    face_box, face_name = detect_face(img)
    # 将检测结果进行展示
    for box, name in zip(face_box, face_name):
        top = box[0] + 3 # 避免图片上的姓名与人脸框重叠
        bottom = box[1]
        left = box[2] + 3 # 避免图片上的姓名与人脸框重叠
        right = box[3]
        # cv2绘制人脸框
        img = cv2.rectangle(img, (left,top), (right, bottom), config['face_box_color'], config['line_thickness'])
        # 图片格式由cv2转至PIL
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        # 向图片上写姓名
        img = add_txt(img, config['font_size'], left, top, name)
        # 将图片转回cv2格式
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

    # 是否保存图片
    if save_img:
        cv2.imwrite(save_path, img)
    # 是否展示图片
    if show:
        cv2.imshow('img',img)
        cv2.waitKey()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    image_path = '000129.jpg'
    main(image_path)

