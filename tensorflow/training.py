import os
import numpy as np
import tensorflow as tf
import input_data
import model
import cv2

N_CLASSES = 4
IMG_H = 128
IMG_W = 128
BATCH_SIZE = 32
CAPACITY = 3000
MAX_STEP = 8000
learning_rate = 0.0001



def run_training():
    train_dir = "./data/TRAIN/"
    logs_train_dir = "./logs/"

    train, train_label = input_data.get_files(train_dir)

    train_batch, train_label_batch = input_data.get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)


    train_logits = model.inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss = model.losses(train_logits,train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)


    summary_op = tf.summary.merge_all()
    sess = tf.Session()

    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc= sess.run([train_op, train_loss, train_acc])

            if step % 100 == 0:
                print('Step:', step, 'train loss:', tra_loss, 'train accuracy:', tra_acc)
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if tra_acc > 0.95 and step>6000:

                checkpoint_path = os.path.join(logs_train_dir, "model")
                saver.save(sess, checkpoint_path, global_step=step)
                print("train success!")
                print('Step:', step, 'train loss:', tra_loss, 'train accuracy:', tra_acc)
                coord.request_stop()
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached.")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# 评估模型


def get_one_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = cv2.imread(img_dir)
    image = cv2.resize(image, (IMG_H, IMG_W), interpolation=cv2.INTER_CUBIC)
    image = np.array(image)
    return image,img_dir

def get_one_origin_image(train):
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    origin_image = cv2.imread(img_dir)
    cv2.imshow('original',origin_image)
######################原始图像处理
    for l in range((origin_image.shape)[0]): #去除黑边
        for w in range((origin_image.shape)[1]):
            if (origin_image[l][w].sum() < 100):
                origin_image[l][w] = [210, 209, 200]
    hsv_img = cv2.cvtColor(origin_image, cv2.COLOR_BGR2HSV)  # HSV空间
    h, s, v = cv2.split(hsv_img)
    ret, binary = cv2.threshold(s, 90, 255, cv2.THRESH_BINARY)  # 选取s通道进行二值化
    last_image = cv2.medianBlur(binary, 5)  # 核越大，能去除的噪点越大
    last_image = cv2.merge([last_image,last_image,last_image])
#######################
    cv2.imshow('last_image',last_image)
    last_image = cv2.resize(last_image, (IMG_H, IMG_W), interpolation=cv2.INTER_CUBIC)
    last_image = np.array(last_image)
    return last_image,img_dir


def evaluate_one_image():

    #测试原始图片
    train_dir = "./data/ORIGIN_TEST/"
    train, train_label = input_data.get_files(train_dir)
    image_array, image_dir = get_one_origin_image(train)

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        image = tf.cast(image_array, tf.float32)
        image = tf.reshape(image, [1, IMG_H, IMG_W, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[IMG_H, IMG_W, 3])

        logs_train_dir = "./logs/"
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loading success, global_step is %s" % global_step)
            else:
                print("No checkpoint file found")

            print('The test picture is :', image_dir)
            prediction = sess.run(logit, feed_dict={x: image_array})
            print(prediction)
            max_index = np.argmax(prediction)
            if max_index == 0:
                # print("This is a eosinophil cell with possibility %.6f" % prediction[:, 0])
                print("This is a eosinophil cell")
            elif max_index==1:
                # print("This is a lymphocyte cell with possibility %.6f" % prediction[:, 1])
                print("This is a lymphocyte cell")
            elif max_index==2:
                # print("This is a monocyte cell with possibility %.6f" % prediction[:, 1])
                print("This is a monocyte cell")

            elif max_index==3:
                # print("This is a neutrophil cell with possibility %.6f" % prediction[:, 1])
                print("This is a neutrophil cell")
            else:
                print('can not recognize the cell')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


run_training()
evaluate_one_image()
