# Face_Recognition
# 配置及环境
## 硬件环境
使用了1台Linux服务器以及一台MacBook Pro，Linux服务器带有4个RTX2080ti GPU。
注：本项目硬件环境需求很低，单用MacBook Pro也可完成。
## 依赖项
TensorFlow-gpu==1.13.1
Keras==2.2.4
OpenCV-python==4.2.0.32
Dlib==1.19.0
# 代码运行
## 文件结构
根目录如图所示：
 
Files文件夹如图所示：
 
所有可运行的文件都放在根目录下就好了，Files主要是存储数据以及训练好的模型权重文件。各个Python文件功能如下：
	Config.py：存储各种参数、路径设置；
	Get_Face_Image.py：用于数据生成（原始数据集可能是半身照，用这个生成指定人数、图数、尺寸的人脸照），这个文件的参数要在文件内设置，而不是Config，因为他自己是个独立的功能，它的输入包括：原始文件夹路径、输出文件夹路径、单人最少图片数、单人最大图片数、图片尺寸、人数；
	Get_Data.py：是模型load数据用的，因为本项目的数据量还算小，所以图省事就没有写data_generator；
	Read_Data_Image.py：就是一些小工具，帮忙读数据或者图片的；
	Train.py：搭建模型和训练模型和评估模型都在这里，可以选择要干啥操作，在两种模型之间切换也是通过这里来调整。
	Show.py：模型训练完成后，可用这个文件来进行展示，因为训练数据集是公开数据集，所以也就没有做电脑摄像头摄像展示的功能，只能单张图片的预测（在大图中标记人脸框和人名，因数据集的问题，现有人名实为编号，不过反正都是外国人写了人名你也不认识），如果需要摄像头展示，可去参考的github中找一下。
其他几个文件功能如下：
	simhei.ttf：字体文件，PIL库网图片上写人名用的，OpenCV可以写英文，但是写中文就会乱码，所以写字这块就使用了PIL。
	.h5：这个结尾的就是模型的训练权重。可以load并进行预测。
	name_list.csv：训练模型的时候，类其实是用的阿拉伯数字表示的，用这个文件可以把模型预测的结果与人名对应回来（通过index确定位置即可）。
	.jpg：就是测试图片了。
6.2.	数据结构
原始数据和处理后的数据都推荐放在一个单独的文件夹里，文件结构均为：总文件夹下有n个人名文件夹，每个人名文件夹下是这个人的图片。以下是一个样例，总文件夹的位置可以自己定。
 
6.3.	数据生成
数据生成是一个单独的功能，修改数据路径及相关参数设置后，运行Get_Face_Image.py文件即可（这也是第一个要运行的文件）。
数据生成的逻辑其实很简单：
1.	文件路径递归：由总文件夹找到人名文件夹再找到单张图片；
2.	使用DLIB对单张图片进行人脸检测；
3.	获取检测到的人脸截图，resize到指定大小后，暂存在内存里；
4.	多次检测之后，若内存里的图片数达到设置的最少单人图片数要求，就把图片保存至指定位置；
5.	如此往复，直到满足设置的人数、图数要求。
6.4.	模型搭建与训练
模型搭建与训练过程全部在Train.py文件中进行。
根据这里写的顺序，依次进行：从磁盘中读取文件、定义模型大类、为实例化数据、建立模型结构（基本CNN或mobile net）、训练模型、评估模型。
6.4.1.	read_trainData函数
这个函数本质上来说就是把Get_Data.py生成的数据集读到类里来，Get_Data的作用其实是从处理好的数据中读取所有的图片、标签、计数（也就是分类数）、人名列表，然后对数据进行训练、验证集拆分，拆分之后再对图片做reshape（加上batch这一维度）、resize（确定输入形状）、归一化（方便模型收敛）等操作，再把label做个one_hot，最后返回出来。
6.4.2.	build_basic_cnn_model函数
这个函数中搭建了基本CNN结构的模型，逻辑是：有训练好的模型就用训练好的那个，没有就新建一个。
模型其实在上面模型介绍章节已做说明，而且基础CNN模型千变万化也没什么明确的规则，可根据自己喜好随便修改，在这里就再详细介绍了。
6.4.3.	build_MobileNet_model函数
这个函数就是搭建mobile net模型用的，逻辑与上一个模型相同，且也在模型介绍章节做了详尽的介绍，从代码层面来说：
 
1.	base_model就是直接利用keras的api定义一个mobile net网络结构；
2.	使用model.input可获取模型的输入；
3.	使用model.layers[-1].output可获取模型最后一层的输出，并暂时定义为x；
4.	然后将x传入一个dense层，本层的形状就是数据要分的类数；
5.	再后就是softmax激活，这个也是此类人脸识别的核心，即为逐渐挖掘图片特征信息后，直接进行分类，谁大就是谁。（另一种人脸识别方式为相似度比对，模型只输出获取的人脸特征向量，再对向量之间的相似度进行比较，也就是一个看看这张图和谁更像的过程）；
6.	激活完成后重新把输出输出包装成一个keras的Model；
7.	再传给大类就好。
6.4.4.	train_model函数
这个函数包含了模型编译及训练两部分。
使用model.compile()为模型编译优化器、损失等参数；
使用model.fit()进行模型训练。
6.4.5.	evaluate_model函数
这个函数的核心就是model.evaluate，可输出模型的损失和精确度。
这里有两个小问题：
1.	在代码逻辑中，模型训练后的那一瞬间，在磁盘中保存了模型权重，在内存中还存储这model这个东西，如果直接跟model.evaluate这个api，会直接使用内存中的model，可是我们模型训练的最后一个epoch不一定是精度最高的那个，在ModelCheckpoint中我也设置了只保存最佳模型，所以这里一定要先从磁盘中load一次模型，在进行评估，这样才是真正的精确度最高的那个模型。
2.	Keras中计算acc好像是基于二分类计算的，这里直接用不一定完全合适，我也尝试写了计算召回率、f1_score的函数塞到callbacks里，但是可能读epoch之后数据那里有点问题，没能成功运行，这里可之后再进行更新迭代。
6.5.	效果展示
效果展示功能都在Show.py文件中。我在Show函数中写了大量的批注，用的逻辑都超级简单，很好看懂，就不再多做解释了，唯一一个扼要注意的点就是，所有带‘load’或者说导入属性的操作，能少做就少做，如果有循环，就放在循环的最外面，这样能够大大提高程序运行速度。
因使用的训练集是公开数据集，不存在通过摄像头看自己是谁的情况，所以没有使用调取摄像头的api。如果使用了自己的数据集，想看看自己是不是自己的话，可以去我参考的那个github中查阅相关函数。
