from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras.layers.pooling import MaxPooling2D
import keras.layers
from keras.models import Model
from keras.optimizers import Adam, TFOptimizer
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import numpy as np


SCREEN_H = 168
SCREEN_W = 168

DECAY_STEPS = 10000
DECAY_RATE = 0.9


class InverseNet:

    def __init__(self, init_lr=5e-4):
        self.init_lr = init_lr
        self.batch_size = 64
        self.reg_coef = 1e-6
        self.decay = 1e-6

    def build_net(self):
        ob_pre_ctrl = Input(shape=[SCREEN_H, SCREEN_W, 3], dtype="float32")
        ob_post_ctrl = Input(shape=[SCREEN_H, SCREEN_W, 3], dtype="float32")

        norm_layer = Lambda(lambda x: x / 255.0)
        x_pre_ctrl = norm_layer(ob_pre_ctrl)
        x_post_ctrl = norm_layer(ob_post_ctrl)
        
        conv1 = Conv2D(32,
                       kernel_size=(9, 9),
                       strides=(3, 3),
                       activation="relu",
                       kernel_regularizer=l2(self.reg_coef))
        x_pre_ctrl = conv1(x_pre_ctrl)
        x_post_ctrl = conv1(x_post_ctrl)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        x_pre_ctrl = maxpool1(x_pre_ctrl)
        x_post_ctrl = maxpool1(x_post_ctrl)

        conv2 = Conv2D(64,
                       kernel_size=(5, 5),
                       strides=(2, 2),
                       activation="relu",
                       kernel_regularizer=l2(self.reg_coef))
        x_pre_ctrl = conv2(x_pre_ctrl)
        x_post_ctrl = conv2(x_post_ctrl)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        x_pre_ctrl = maxpool2(x_pre_ctrl)
        x_post_ctrl = maxpool2(x_post_ctrl)

        flatten = Flatten()
        x_pre_ctrl = flatten(x_pre_ctrl)
        x_post_ctrl = flatten(x_post_ctrl)

        x = keras.layers.concatenate([x_pre_ctrl, x_post_ctrl])
        x = Dense(512, activation="relu", kernel_regularizer=l2(self.reg_coef))(x)
#         x = Dense(1024, activation="relu", kernel_regularizer=l2(self.reg_coef))(x)
        est_ctrl = Dense(2, kernel_regularizer=l2(self.reg_coef))(x)

        self.model = Model([ob_pre_ctrl, ob_post_ctrl], [est_ctrl])
        
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(learning_rate=self.init_lr,
                                        global_step=global_step,
                                        decay_steps=DECAY_STEPS,
                                        decay_rate=DECAY_RATE,
                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer = TFOptimizer(optimizer)
#         optimizer = Adam(self.init_lr, decay=self.decay)
        
        def inv_loss(y_true, y_pred):
            return K.mean(K.sum(K.square(y_true - y_pred), axis=-1))
        
        self.model.compile(optimizer, loss=inv_loss)
    
    def train(self, train_pre_ctrl, train_post_ctrl, train_ctrl,
              valid_pre_ctrl, valid_post_ctrl, valid_ctrl, epochs=64):
        
        train_hist = self.model.fit(x=[train_pre_ctrl, train_post_ctrl],
                                    y=[train_ctrl],
                                    batch_size=self.batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=[[valid_pre_ctrl, valid_post_ctrl], [valid_ctrl]],
                                    shuffle=True,
                                    callbacks=[ModelCheckpoint(filepath="invnet.h5",
                                                               verbose=0,
                                                               monitor="val_loss",
                                                               mode="min",
                                                               save_best_only=True,
                                                               save_weights_only=True)])

