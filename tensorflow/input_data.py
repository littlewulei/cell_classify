import tensorflow as tf
import numpy as np
import os
# 获取文件路径和标签
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 乱序后的图片和标签
    eosinophil=[]
    lymphocyte=[]
    monocyte=[]
    neutrophil=[]

    label_eosinophil=[]
    label_lymphocyte=[]
    label_monocyte=[]
    label_neutrophil=[]

    # 载入数据路径并写入标签值
    class_file = os.listdir(file_dir)#文件名列表并不是有序的
    for i in class_file:
        if i=='eosinophil':
            cell_file_dir = os.listdir(file_dir+'eosinophil/')
            print()
            for j in cell_file_dir:
                eosinophil.append(file_dir+'eosinophil/'+j)
                label_eosinophil.append(0)

        elif i=='lymphocyte':
            cell_file_dir = os.listdir(file_dir + 'lymphocyte/')
            for j in cell_file_dir:
                lymphocyte.append(file_dir+'lymphocyte/'+j)
                label_lymphocyte.append(1)

        elif i=='monocyte':
            cell_file_dir = os.listdir(file_dir + 'monocyte/')
            for j in cell_file_dir:
                monocyte.append(file_dir+'monocyte/' + j)
                label_monocyte.append(2)

        elif i=='neutrophil':
            cell_file_dir = os.listdir(file_dir + 'neutrophil/')
            for j in cell_file_dir:
                neutrophil.append(file_dir+'neutrophil/' + j)
                label_neutrophil.append(3)
        else:
            pass

    print("There are %d eosinophil\nThere are %d lymphocyte" % (len(eosinophil), len(lymphocyte)))
    print("There are %d monocyte\nThere are %d neutrophil" % (len(monocyte), len(neutrophil)))
    # print(eosinophil[-5:-1],label_eosinophil[-5:-1])


    # 打乱文件顺序
    # image_list = np.hstack((eosinophil, lymphocyte,monocyte,neutrophil))
    image_list=np.concatenate((eosinophil, lymphocyte, monocyte, neutrophil), axis=0)#合并
    # print(image_list[-5:-1])
    # label_list = np.hstack((label_eosinophil, label_lymphocyte,label_monocyte,label_neutrophil))
    label_list = np.concatenate((label_eosinophil, label_lymphocyte,label_monocyte,label_neutrophil), axis=0)#合并
    # print(label_list[-5:-1])
    temp = np.array([image_list, label_list])
    temp = temp.transpose()     # 转置
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list
# 生成相同大小的批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列
    input_queue = tf.train.slice_input_producer([image, label])

    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 统一图片大小
    # 视频方法
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 我的方法
    image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)   # 标准化数据
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   # 线程
                                              capacity=capacity)



    return image_batch, label_batch

# get_files('./data/TRAIN/')
# get_files('./data/TEST/')