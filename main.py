import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from glob import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import  Activation
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"




import tensorflow as tf
from tensorflow.keras.metrics import Recall, Precision

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from metrics import dice_loss, dice_coef, iou
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from model import deeplabv3_plus
import CloudNet

from tensorflow.keras import metrics
import datetime

import gc
H = 384
W = 384
#https://discuss.tensorflow.org/t/model-is-not-learning/5585
""" Creating a directory """
def create_dir(path):#This creates a new director if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):#This suffles the datset
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):#This is used to load the images and masks
    x = sorted(glob(os.path.join(path, "img", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y

def read_image(path):#This read the images from the dataset
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):#This reads the masks from the dataset
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):#This gets all the images and masks from the dataset
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=1):#This creates a tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset











if __name__ == "__main__":#This is ran when the file is the main
    """ Seeding """
    
    print("Loading")
    """ Directory for storing files """
    create_dir("files/test")

    """ Hyperparameters """
    batch_size = 1
    lr = 1e-4
    num_epochs = 2
    model_path = os.path.join("files/test", "model.h5")
    csv_path = os.path.join("files/test", "data.csv")

    """ Dataset """
    #dataset_path = "dataset/training"
    train_path = os.path.join("dataset", "training")
    valid_path = os.path.join("dataset", "test")

    train_x, train_y = load_data(train_path)#Loads the training dataset
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)#Loads the validation dataset

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)#Creates the tensorflow training datset
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)#Creates the tensorflow validation datset
    train_dataset = train_dataset.shuffle(buffer_size=64)
    train_dataset = train_dataset.repeat()
    valid_dataset = valid_dataset.shuffle(buffer_size=64)
    valid_dataset = valid_dataset.repeat()





def double_conv_block(x, n_filters):#This is a double converlational block 

    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

    return x

def downsample_block(x, n_filters):#This is the down sampling block
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):#This is the upsampling block
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate 
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x
#https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET/blob/main/model.py
def attentionblock(x,gating,inter_shape):#This is the attention gate
    shape_x = keras.backend.int_shape(x)
    shape_g = keras.backend.int_shape(gating)
    theta_x = layers.Conv2D(inter_shape,(1,1), strides=(2,2), padding = 'same')(x)
    phi_g = layers.Conv2D(inter_shape,(1,1), padding= 'same')(gating)

    concat_xg = layers.add([phi_g,theta_x])

    act_xg = layers.Activation('relu')(concat_xg)

    psi = layers.Conv2D(1, (1,1), padding='same')(act_xg)

    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = keras.backend.int_shape(sigmoid_xg)

    upsample_psi = layers.UpSampling2D(size=(shape_x[1]//shape_sigmoid[1], shape_x[2]//shape_sigmoid[2]))(sigmoid_xg)

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1,1), padding = 'same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def gating_signal(input, out_size, batch_norm=False):#This gets the signal from the lower layer
   
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
def first_build_unet_model(past, ina):#This is for the first unet in a AMEL
  
  input_layer = tf.keras.layers.concatenate([ina,past.outputs[0]],3)
  return build_unet_model(input_layer)
def other_build_unet_model(past, ina):#This is for the rest unet in a AMEL
  input_layer = tf.keras.layers.concatenate([ina,past.outputs[0]])
  return build_unet_model(input_layer)

def normal_unet_model(past, ina):#This is for the head of each part of a amel u-net
  input_layer = tf.keras.layers.concatenate([ina,past.outputs[0]])
  return normal_build_unet_model_test(input_layer)
def build_unet_model(input_layer):
#https://github.com/bnsreenu/python_for_microscopists/blob/072ef815f325f56a59a0e88369c6b2d6e7ef25cc/224_225_226_models.py
    
    # 1 - downsample
    #input_layer = tf.keras.layers.concatenate([ina,past.outputs], name = "a")
    f1, p1 = downsample_block(input_layer, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    get1 = gating_signal(bottleneck,512,False)
    at1 = attentionblock(f4,get1,512)
    
    u6 = layers.concatenate([u6, at1],axis= 3)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    get2 = gating_signal(u6,256,False)
    at2 = attentionblock(f3,get2,256)
    u7 = layers.concatenate([u7, at2],axis=3)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    get3 = gating_signal(u7,128,False)
    at3 = attentionblock(f2,get3,128)
    u8 = layers.concatenate([u8, at3],axis=3)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    get4 = gating_signal(u8,64,False)
    at4 = attentionblock(f1,get4,64)
    u9 = layers.concatenate([u9, at4],axis=3)
    
    
    outputs2 = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    outputs2 = Activation("sigmoid")(outputs2)
    
    return outputs2 

def build_unet_model_test(input_layer):#This is a attention u-net
#https://github.com/bnsreenu/python_for_microscopists/blob/072ef815f325f56a59a0e88369c6b2d6e7ef25cc/224_225_226_models.py
    
    # encoder: contracting path - downsample
    # 1 - downsample
    #input_layer = tf.keras.layers.concatenate([ina,past.outputs], name = "a")
    f1, p1 = downsample_block(input_layer, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    get1 = gating_signal(bottleneck,512,False)
    at1 = attentionblock(f4,get1,512)
    
    u6 = layers.concatenate([u6, at1],axis= 3)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    get2 = gating_signal(u6,256,False)
    at2 = attentionblock(f3,get2,256)
    u7 = layers.concatenate([u7, at2],axis=3)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    get3 = gating_signal(u7,128,False)
    at3 = attentionblock(f2,get3,128)
    u8 = layers.concatenate([u8, at3],axis=3)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    get4 = gating_signal(u8,64,False)
    at4 = attentionblock(f1,get4,64)
    u9 = layers.concatenate([u9, at4],axis=3)
    
    
    outputs2 = layers.Conv2D(1, 1, padding="same")(u9)
    
    outputs2 = Activation("sigmoid")(outputs2)
   
    return outputs2 

def normal_build_unet_model_test(input_layer):#this is a u-net
  #https://github.com/bnsreenu/python_for_microscopists/blob/072ef815f325f56a59a0e88369c6b2d6e7ef25cc/224_225_226_models.py
  
  # encoder: contracting path - downsample
  # 1 - downsample
  
  f1, p1 = downsample_block(input_layer, 64)
  # 2 - downsample
  f2, p2 = downsample_block(p1, 128)
  # 3 - downsample
  f3, p3 = downsample_block(p2, 256)
  # 4 - downsample
  f4, p4 = downsample_block(p3, 512)

  # 5 - bottleneck
  bottleneck = double_conv_block(p4, 1024)

  # decoder: expanding path - upsample
  # 6 - upsample
  u6 = upsample_block(bottleneck, f4, 512)
  
  # 7 - upsample
  u7 = upsample_block(u6, f3, 256)
  
  # 8 - upsample
  u8 = upsample_block(u7, f2, 128)
  
  # 9 - upsample
  u9 = upsample_block(u8, f1, 64)
  
  outputs2 = layers.Conv2D(1, 1, padding="same")(u9)
  
  
  outputs2 = Activation("sigmoid")(outputs2)
  
  return outputs2 


def finalModel (inpu):#this is the head AMEL of adau
  cov = layers.Conv2D(3,(3,3), padding= 'same', activation = "relu")(inpu)
  cov = layers.Conv2D(3,(3,3), padding= 'same', activation = "relu")(cov)
  cov2 = layers.Conv2D(1, 1)(cov)
  
  output = Activation("sigmoid")(cov2)
 
  return output




def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]



def aeml(input_layer,unet1,unet2,unet3):#This is the amel for the amel models
  con = layers.concatenate([unet1.outputs[0],unet2.outputs[0],unet3.outputs[0]])
  #cov1 = layers.Conv2D(3, 1, padding="same", activation = "softmax")(con)
  cov = layers.Conv2D(3,(3,3), padding= 'same', activation = "relu")(con)
  cov = layers.Conv2D(3,(3,3), padding= 'same', activation = "relu")(cov)
  cov2 = layers.Conv2D(1, 1)(cov)
  #cov2 = Activation("relu")(cov2)
  cov2 = Activation("sigmoid")(cov2)
  #output = tf.keras.activations.softmax(cov2)
  final =tf.keras.Model(inputs = input_layer , outputs = cov2, name = 'Final')
  return final
input_layer = tf.keras.layers.Input(shape=(384, 384, 3), name = 'input_layer first')



mode = input("1 Train ADAU, 2 Evaluate models, 3 Train testing models: ")
print(mode)
def imageProc(a):#This pocesses the predition images
  a = cv2.resize(a, (w, h))
  a = np.expand_dims(a, axis=-1)
  a = np.where(a>0.5,1,0)#This is the cut off function
  a = a *255
  a = np.stack((a,a,a),axis=2)
  a = a[:,:,:,0]
  return a
if mode == "1":#This section is for training the adau
    print("Training..")
    
    NUM_EPOCHS = 100

    TRAIN_LENGTH = 200
    STEPS_PER_EPOCH = TRAIN_LENGTH // 2

    VAL_SUBSPLITS = 2
    TEST_LENTH = 200
    VALIDATION_STEPS = TEST_LENTH // 2 // VAL_SUBSPLITS
    
    
    
    
    deeplab = deeplabv3_plus((384,384,3))
    deeplab.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
    checkpoint_path_deeplab = "finaldeeplabv3plus.ckpt"
    callbacks = [
      ModelCheckpoint(checkpoint_path_deeplab, verbose=1, save_best_only=True, save_weights_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
      CSVLogger(csv_path),
      TensorBoard(),
      EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
    if os.path.exists(checkpoint_path_deeplab + ".index"): 
      deeplab.load_weights(checkpoint_path_deeplab)
    
    his1 = deeplab.fit(
      train_dataset,
      epochs=NUM_EPOCHS,
      steps_per_epoch= STEPS_PER_EPOCH,
      validation_data=valid_dataset,
      validation_steps= VALIDATION_STEPS,
      callbacks=callbacks
    )
    
    checkpoint_pathunet1 = "finalunet1.ckpt"
    cp_callback1 = [
      ModelCheckpoint(checkpoint_pathunet1, verbose=1, save_best_only=True, save_weights_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
      CSVLogger(csv_path),
      TensorBoard(),
      EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
    tf.keras.utils.plot_model(deeplab, show_shapes=True,to_file="tdeeplab.png",show_layer_activations=True,show_dtype=True,dpi=100)

    deeplab.load_weights(checkpoint_path_deeplab)
    deeplab.trainable = False
    tmp = train_dataset.take(1)
    
    input_layer = tf.keras.layers.Input(shape=(384, 384, 3), name = 'input_layer come on what is wrong')
    unet1m = first_build_unet_model(deeplab,deeplab.input)
    unet1 = tf.keras.Model(inputs = deeplab.input, outputs = unet1m)
    unet1.compile(optimizer=tf.keras.optimizers.SGD(),
                loss="sparse_categorical_crossentropy",
                metrics="accuracy")
    tf.keras.utils.plot_model(unet1, show_shapes=True,to_file="tUnet1.png",show_layer_activations=True,show_dtype=True,dpi=100)
    if os.path.exists(checkpoint_pathunet1+ ".index"):
      unet1.load_weights(checkpoint_pathunet1)
    his2 = unet1.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=valid_dataset, callbacks=[cp_callback1])
    
    checkpoint_pathunet2 = "finalunet2.ckpt"
    cp_callback1 = [
      ModelCheckpoint(checkpoint_pathunet2, verbose=1, save_best_only=True, save_weights_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
      CSVLogger(csv_path),
      TensorBoard(),
      EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
    deeplab.trainable = False
    unet1.trainable = False
    unet1.load_weights(checkpoint_pathunet1)
    unet2m = other_build_unet_model(unet1,deeplab.input)
    unet2 = tf.keras.Model(inputs = deeplab.input, outputs = unet2m)
    unet2.compile(optimizer=tf.keras.optimizers.SGD(),
                loss="sparse_categorical_crossentropy",
                metrics="accuracy")
    if os.path.exists(checkpoint_pathunet2+ ".index"):
      unet2.load_weights(checkpoint_pathunet2)
    tf.keras.utils.plot_model(unet2, show_shapes=True,to_file="tUnet.png",show_layer_activations=True,show_dtype=True,dpi=100)
    his3 = unet2.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=valid_dataset, callbacks=[cp_callback1])
    
    checkpoint_pathunet3 = "finalunet3.ckpt"
    cp_callback1 = [
      ModelCheckpoint(checkpoint_pathunet3, verbose=1, save_best_only=True, save_weights_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
      CSVLogger(csv_path),
      TensorBoard(),
      EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
    deeplab.trainable = False
    unet1.trainable = False
    unet2.trainable = False
    unet2.load_weights(checkpoint_pathunet2)
    unet3m = other_build_unet_model(unet2,deeplab.input)
    unet3 = tf.keras.Model(inputs = deeplab.input, outputs = unet3m)
    unet3.compile(optimizer=tf.keras.optimizers.SGD(),
                loss="sparse_categorical_crossentropy",
                metrics="accuracy")
    if os.path.exists(checkpoint_pathunet3+ ".index"):
      unet3.load_weights(checkpoint_pathunet3)
    tf.keras.utils.plot_model(unet3, show_shapes=True,to_file="tUnet.png",show_layer_activations=True,show_dtype=True,dpi=100)

    his4 = unet3.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=valid_dataset, callbacks=[cp_callback1])
    count = 0
    deeplab.trainable = False
    unet1.trainable = False
    unet2.trainable = False
    unet3.trainable = False
    unet3.load_weights(checkpoint_pathunet3)
    fin = tf.keras.layers.concatenate([deeplab.outputs[0],unet1.outputs[0],unet2.outputs[0],unet3.outputs[0]],name="concatenated_layer")
    final = finalModel(fin)
    adau = tf.keras.Model(inputs = deeplab.input, outputs = final, name = 'Final')
    tf.keras.utils.plot_model(adau, show_shapes=True,to_file="tFinal.png",show_layer_activations=True,show_dtype=True,dpi=100)
    checkpoint_pathfinal = "finalfinal2a.ckpt"
    cp_callback1 = [
      ModelCheckpoint(checkpoint_pathfinal, verbose=1, save_best_only=True, save_weights_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
      CSVLogger(csv_path),
      TensorBoard(),
      EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
    if os.path.exists(checkpoint_pathfinal+ ".index"):
      adau.load_weights(checkpoint_pathfinal)
    
    adau.compile(loss=[dice_loss], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print("Final")
    his5 = adau.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=valid_dataset, callbacks=[cp_callback1, tensorboard_callback])
    
    
    
    
    t = adau.evaluate(valid_dataset, steps = 1488)
    print(t)
   
    
    
    

    
elif mode == "2":#This section evaluates each model
  print("Evaluate")
  num_epochs = 100
  Atenunet = build_unet_model_test(input_layer)
  checkpoint_path = "testAtenunet.ckpt"
  cp_callback1 = [
      ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
      CSVLogger(csv_path),
      TensorBoard(),
      EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
  
  Atenunet = tf.keras.Model(inputs = input_layer, outputs = Atenunet)
  Atenunet.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(), metrics.MeanAbsoluteError(), "accuracy"])
  Atenunet.load_weights(checkpoint_path).expect_partial()
  print("AtenUnet")



  


  deeplab = deeplabv3_plus((384,384,3))
  deeplab.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(), metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_deeplab = "testingdeeplabv3plus.ckpt"
  
  deeplab.load_weights(checkpoint_path_deeplab).expect_partial()
  print("deeplab")
  
  
  unet1 = build_unet_model_test(input_layer)
  unet1 = tf.keras.Model(inputs = input_layer, outputs = unet1)
  unet1.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(), metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_deeplab = "testingunet1.ckpt"
  print("Unet1")

  ina = tf.keras.layers.concatenate([input_layer,unet1.outputs[0]],3)

  unet1.load_weights(checkpoint_path_deeplab).expect_partial()
  unet1.trainable = False
  unet2 = other_build_unet_model(unet1, ina)
  unet2 = tf.keras.Model(inputs = input_layer, outputs = unet2)

  unet2.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(), metrics.MeanAbsoluteError(),"accuracy"])
  checkpoint_path_deeplab = "testingunet2.ckpt"
 
  print("Unet2")
  
 

  unet2.load_weights(checkpoint_path_deeplab).expect_partial()
  unet2.trainable = False

  ina = tf.keras.layers.concatenate([input_layer,unet2.outputs[0]],3)

  unet3 = other_build_unet_model(unet2, ina)
  unet3 = tf.keras.Model(inputs = input_layer, outputs = unet3)
  unet3.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(), metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_deeplab = "testingunet3.ckpt"
  
  print("Unet3")
 
  

  unet3.load_weights(checkpoint_path_deeplab).expect_partial()
  unet3.trainable = False

  amelUnet = aeml(input_layer, unet1, unet2, unet3)
  
  amelUnet.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(), metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_amel = "testingamelAtentionUnet.ckpt"
  
  amelUnet.load_weights(checkpoint_path_amel).expect_partial()
  tf.keras.utils.plot_model(amelUnet, show_shapes=True,to_file="testingAmel.png",show_layer_activations=True,show_dtype=True,dpi=100)
  print("amelUnet")
  

  nunet = normal_build_unet_model_test(input_layer)
  nunet = tf.keras.Model(inputs = input_layer, outputs = nunet)

  nunet.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(), metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_deeplab = "testingnormalUnet.ckpt"
  
  nunet.load_weights(checkpoint_path_deeplab).expect_partial()
  tf.keras.utils.plot_model(nunet, show_shapes=True,to_file="testingnormalUnet.png",show_layer_activations=True,show_dtype=True,dpi=100)
  print("normal Unet")
  
  
  deeplab2 = deeplabv3_plus((384,384,3))
  deeplab2.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_deeplab = "finaldeeplabv3plus.ckpt"
  deeplab2.load_weights(checkpoint_path_deeplab).expect_partial()
  print("adau deeplab")
  checkpoint_pathunet1 = "finalunet1.ckpt"
  unet1m = first_build_unet_model(deeplab2,deeplab2.input)
  unet1a = tf.keras.Model(inputs = deeplab2.input, outputs = unet1m)
  unet1a.compile(optimizer=tf.keras.optimizers.SGD(),
              loss="sparse_categorical_crossentropy",
              metrics="accuracy")
  unet1a.load_weights(checkpoint_pathunet1).expect_partial()
  print("adau unet1")
  checkpoint_pathunet2 = "finalunet2.ckpt"
  unet2m = other_build_unet_model(unet1a,deeplab2.input)
  unet2a = tf.keras.Model(inputs = deeplab2.input, outputs = unet2m)
  unet2a.compile(optimizer=tf.keras.optimizers.SGD(),
              loss="sparse_categorical_crossentropy",
              metrics="accuracy")
  unet2a.load_weights(checkpoint_pathunet2).expect_partial()
  print("adau unet2")
  
  checkpoint_pathunet3 = "finalunet3.ckpt"
  unet3m = other_build_unet_model(unet2a,deeplab2.input)
  unet3a = tf.keras.Model(inputs = deeplab2.input, outputs = unet3m)
  unet3a.compile(optimizer=tf.keras.optimizers.SGD(),
              loss="sparse_categorical_crossentropy",
              metrics="accuracy")
  unet3a.load_weights(checkpoint_pathunet3).expect_partial()
  print("adau unet3")
  checkpoint_pathfinal = "finalfinal2a.ckpt"
  fin = tf.keras.layers.concatenate([deeplab2.outputs[0],unet1a.outputs[0],unet2a.outputs[0],unet3a.outputs[0]],name="concatenated_layer")
  final = finalModel(fin)
  adau = tf.keras.Model(inputs = deeplab2.input, outputs = final, name = 'Final')
  adau.compile(loss=[dice_loss], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  adau.load_weights(checkpoint_pathfinal).expect_partial().expect_partial()
  print("adau")
  tf.keras.utils.plot_model(adau, show_shapes=True,to_file="ADAU.png",show_layer_activations=True,show_dtype=True,dpi=100)
  
  
  
  unet1 = normal_build_unet_model_test(input_layer)
  unet1 = tf.keras.Model(inputs = input_layer, outputs = unet1)
  unet1.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_unet1 = "testingnormalUnet.ckpt"
 
  print("Unet1n")
  unet1.load_weights(checkpoint_path_unet1).expect_partial()
  
  ina = tf.keras.layers.concatenate([input_layer,unet1.outputs[0]],3)

  unet1.load_weights(checkpoint_path_unet1).expect_partial()
  unet1.trainable = False
  unet2 = normal_unet_model(unet1, ina)
  unet2 = tf.keras.Model(inputs = input_layer, outputs = unet2)

  unet2.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_unet2 = "testingnormalunet2.ckpt"
  
  print("Unet2n")
  tf.keras.utils.plot_model(unet2, show_shapes=True,to_file="testingAmel.png",show_layer_activations=True,show_dtype=True,dpi=100)
  unet2.load_weights(checkpoint_path_unet2).expect_partial()
  
 

  unet2.load_weights(checkpoint_path_unet2).expect_partial()
  unet2.trainable = False

  ina = tf.keras.layers.concatenate([input_layer,unet2.outputs[0]],3)

  unet3 = normal_unet_model(unet2, ina)
  unet3 = tf.keras.Model(inputs = input_layer, outputs = unet3)
  unet3.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_unet3 = "testingnormalunet3.ckpt"
  
  print("Unet3n")
  unet3.load_weights(checkpoint_path_unet3).expect_partial()
  
  

  unet3.load_weights(checkpoint_path_unet3).expect_partial()
  unet3.trainable = False

  normalamelUnet = aeml(input_layer, unet1, unet2, unet3)
  
  normalamelUnet.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_amel = "testingamelnormalUnet.ckpt"
  
  normalamelUnet.load_weights(checkpoint_path_amel).expect_partial()
  tf.keras.utils.plot_model(normalamelUnet, show_shapes=True,to_file="testingAmel.png",show_layer_activations=True,show_dtype=True,dpi=100)
  print("amelnormalUnet")
  
  cloudNet = CloudNet.model_arch(input_rows=384,
                                       input_cols=384,
                                       num_of_channels=3,
                                       num_of_classes=1)
  cloudNet.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  
  checkpoint_path_cloudnet = "cloudnet.ckpt"
  cloudNet.load_weights(checkpoint_path_cloudnet).expect_partial()
  
  print("cloudnet")
  print("Attention U-Net evaluation")
  a = Atenunet.evaluate(valid_dataset, steps = 1488)
  print("ADAU evaluation")
  b = adau.evaluate(valid_dataset, steps = 1488)
  print("DeepLabV3 Plus evaluation")
  c = deeplab.evaluate(valid_dataset, steps = 1488)
  print("AMEL U-Net")
  d = amelUnet.evaluate(valid_dataset, steps = 1488)
  print("U-Net evaluation")
  e = nunet.evaluate(valid_dataset, steps = 1488)
  print("AMEL U-Net evaluation")
  f = normalamelUnet.evaluate(valid_dataset, steps = 1488)
  print("Cloud-net evaluation")
  g = cloudNet.evaluate(valid_dataset, steps = 1488)
  print(a)
  print(b)
  print(c)
  print(d)
  print(e)
  print(f)
  ten = []
  cat_images = np.ones((50, 2868, 3)) * 255
  
  data_x = glob("Testingimges/img/*")
  i =0
  for path in data_x:
        """ Extracting name """
        name = path.split("\\")[-1].split(".")[0]
        maskpath = "Testingimges/mask/"+name+".png"
        """ Reading the image """
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = cv2.imread(maskpath, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        y = Atenunet.predict(x)[0]
        y = imageProc(y)
        
        
        

        d = deeplab.predict(x)[0]
        d = imageProc(d)

        a = amelUnet.predict(x)[0]
        a = imageProc(a)

        n = nunet.predict(x)[0]
        n = imageProc(n)

        t = adau.predict(x)[0]
        t = imageProc(t)
        
        an = normalamelUnet.predict(x)[0]
        an = imageProc(an)

        
        line = np.ones((h, 30, 3)) * 255
        tpm = np.concatenate([image, line, t, line, d, line, a, line, n,line,an, line, y], axis=1)
        cat_images = np.concatenate([cat_images, tpm], axis=0)
        
        hline = np.ones((30, 2868, 3)) * 255
        cat_images=np.concatenate([cat_images, hline], axis=0)
        
        
        
        i +=1
  del Atenunet
  del amelUnet
  del nunet
  del adau
  del normalamelUnet
  del deeplab
  gc.collect()
  a = np.ones((50, 828, 3)) * 255
  for path in data_x:
      """ Extracting name """
      name = path.split("\\")[-1].split(".")[0]
      maskpath = "Testingimges/mask/"+name+".png"
      """ Reading the image """
      image = cv2.imread(path, cv2.IMREAD_COLOR)
      mask = cv2.imread(maskpath, cv2.IMREAD_COLOR)
      h, w, _ = image.shape
      x = cv2.resize(image, (W, H))
      x = x/255.0
      x = x.astype(np.float32)
      x = np.expand_dims(x, axis=0)

      """ Prediction """
      
      
      
      

      c = cloudNet.predict(x)[0]
      c = imageProc(c)

      
      line = np.ones((h, 30, 3)) * 255
      tpm = np.concatenate([line, c,line, mask], axis=1)
      a = np.concatenate([a, tpm], axis=0)
      
      hline = np.ones((30, 828, 3)) * 255
      a=np.concatenate([a, hline], axis=0)
      
     
      
      i +=1
  cat_images=np.concatenate([cat_images, a], axis=1)
  cv2.imwrite(f"test_images/maskc/out.png", cat_images)

elif mode == "3":#This trains the testing models
  print("Training testing models")
  num_epochs = 100
  Atenunet = build_unet_model_test(input_layer)
  checkpoint_path = "testAtenunet.ckpt"
  cp_callback1 = [
      ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True),
      ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
      CSVLogger(csv_path),
      TensorBoard(),
      EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]
  
  Atenunet = tf.keras.Model(inputs = input_layer, outputs = Atenunet)
  Atenunet.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  if os.path.exists(checkpoint_path+ ".index"):
    Atenunet.load_weights(checkpoint_path) 
  print("AtenUnet")
  Atenunet.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=cp_callback1
  )



  deeplab = deeplabv3_plus((384,384,3))
  deeplab.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_deeplab = "testingdeeplabv3plus.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_deeplab, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  if os.path.exists(checkpoint_path_deeplab+ ".index"): 
    deeplab.load_weights(checkpoint_path_deeplab)
  print("deeplab")
  deeplab.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
  
  unet1 = build_unet_model_test(input_layer)
  unet1 = tf.keras.Model(inputs = input_layer, outputs = unet1)
  unet1.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_unet1 = "testingunet1.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_unet1, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  print("Unet1")
  if os.path.exists(checkpoint_path_unet1+ ".index"): 
    unet1.load_weights(checkpoint_path_unet1)
  unet1.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
  ina = tf.keras.layers.concatenate([input_layer,unet1.outputs[0]],3)

  unet1.load_weights(checkpoint_path_unet1)
  unet1.trainable = False
  unet2 = other_build_unet_model(unet1, ina)
  unet2 = tf.keras.Model(inputs = input_layer, outputs = unet2)

  unet2.compile(optimizer=tf.keras.optimizers.SGD(),
                loss="sparse_categorical_crossentropy",
                metrics="accuracy")
  tf.keras.utils.plot_model(unet2, show_shapes=True,to_file="testingnormalUnet.png",show_layer_activations=True,show_dtype=True,dpi=100)
  checkpoint_path_unet2 = "testingunet2.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_unet2, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  print("Unet2")
  if os.path.exists(checkpoint_path_unet2+ ".index"):
    unet2.load_weights(checkpoint_path_unet2)
  unet2.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )

  unet2.load_weights(checkpoint_path_unet2)
  unet2.trainable = False

  ina = tf.keras.layers.concatenate([input_layer,unet2.outputs[0]],3)

  unet3 = other_build_unet_model(unet2, ina)
  unet3 = tf.keras.Model(inputs = input_layer, outputs = unet3)
  unet3.compile(optimizer=tf.keras.optimizers.SGD(),
                loss="sparse_categorical_crossentropy",
                metrics="accuracy")
  checkpoint_path_unet3 = "testingunet3.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_unet3, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  print("Unet3")
  if os.path.exists(checkpoint_path_unet3+ ".index"):
    unet3.load_weights(checkpoint_path_unet3)
  unet3.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
  

  unet3.load_weights(checkpoint_path_unet3)
  unet3.trainable = False

  amelUnet = aeml(input_layer, unet1, unet2, unet3)
  
  amelUnet.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_amel = "testingamelAtentionUnet.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_amel, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  
  
  tf.keras.utils.plot_model(amelUnet, show_shapes=True,to_file="testingAmel.png",show_layer_activations=True,show_dtype=True,dpi=100)
  print("amelUnet")
  if os.path.exists(checkpoint_path_amel+ ".index"):
    amelUnet.load_weights(checkpoint_path_amel)
  amelUnet.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )#
  amelUnet.load_weights(checkpoint_path_amel)
  
  nunet = normal_build_unet_model_test(input_layer)
  nunet = tf.keras.Model(inputs = input_layer, outputs = nunet)

  nunet.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_nunet = "testingnormalUnet.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_nunet, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  tf.keras.utils.plot_model(nunet, show_shapes=True,to_file="testingnormalUnet.png",show_layer_activations=True,show_dtype=True,dpi=100)
  print("normal Unet")
  if os.path.exists(checkpoint_path_nunet+ ".index"):
    nunet.load_weights(checkpoint_path_nunet)
  nunet.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
  unet1 = normal_build_unet_model_test(input_layer)
  unet1 = tf.keras.Model(inputs = input_layer, outputs = unet1)
  unet1.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_unet1 = "testingnormalUnet.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_unet1, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  print("Unet1n")
  if os.path.exists(checkpoint_path_unet1+ ".index"):
    unet1.load_weights(checkpoint_path_unet1)
  unet1.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
  ina = tf.keras.layers.concatenate([input_layer,unet1.outputs[0]],3)

  unet1.load_weights(checkpoint_path_unet1)
  unet1.trainable = False
  unet2 = normal_unet_model(unet1, ina)
  unet2 = tf.keras.Model(inputs = input_layer, outputs = unet2)

  unet2.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_unet2 = "testingnormalunet2.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_unet2, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  print("Unet2n")
  tf.keras.utils.plot_model(unet2, show_shapes=True,to_file="testingAmel.png",show_layer_activations=True,show_dtype=True,dpi=100)
  if os.path.exists(checkpoint_path_unet2+ ".index"):
    unet2.load_weights(checkpoint_path_unet2)
  unet2.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
 

  unet2.load_weights(checkpoint_path_unet2)
  unet2.trainable = False

  ina = tf.keras.layers.concatenate([input_layer,unet2.outputs[0]],3)

  unet3 = normal_unet_model(unet2, ina)
  unet3 = tf.keras.Model(inputs = input_layer, outputs = unet3)
  unet3.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_unet3 = "testingnormalunet3.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_unet3, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  print("Unet3n")
  if os.path.exists(checkpoint_path_unet3+ ".index"):
    unet3.load_weights(checkpoint_path_unet3)
  unet3.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
  

  unet3.load_weights(checkpoint_path_unet3)
  unet3.trainable = False

  normalamelUnet = aeml(input_layer, unet1, unet2, unet3)
  
  normalamelUnet.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  checkpoint_path_amel = "testingamelnormalUnet.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_amel, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  if os.path.exists(checkpoint_path_amel+ ".index"):
    normalamelUnet.load_weights(checkpoint_path_amel)
  tf.keras.utils.plot_model(normalamelUnet, show_shapes=True,to_file="testingAmel.png",show_layer_activations=True,show_dtype=True,dpi=100)
  print("amelnormalUnet")
  normalamelUnet.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
  cloudnet = CloudNet.model_arch(input_rows=384,
                                       input_cols=384,
                                       num_of_channels=3,
                                       num_of_classes=1)
  cloudnet.compile(loss=[keras.losses.BinaryCrossentropy()], optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision(),  metrics.MeanAbsoluteError(), "accuracy"])
  tf.keras.utils.plot_model(cloudnet, show_shapes=True,to_file="CloudNet.png",show_layer_activations=True,show_dtype=True,dpi=100)
  checkpoint_path_cloudnet = "cloudnet.ckpt"
  callbacks = [
    ModelCheckpoint(checkpoint_path_cloudnet, verbose=1, save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
  ]
  if os.path.exists(checkpoint_path_cloudnet+ ".index"):
    cloudnet.load_weights(checkpoint_path_cloudnet)
  cloudnet.fit(
    train_dataset,
    epochs=num_epochs,
    steps_per_epoch= 100,
    validation_data=valid_dataset,
    validation_steps= 100,
    callbacks=callbacks
  )
