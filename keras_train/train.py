import numpy as np
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
import keras.layers
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import WeightRegularizer

SCREEN_H = 168
SCREEN_W = 168


class InverseNet:

    def __init__(self):
        self.init_lr = 1e-4
        self.batch_size = 64
        self.reg_coef = 1e-5
        self.decay = 1e-5

    def build_net(self):
        print "Start building net ..."
        ob_pre_ctrl = Input(shape=[SCREEN_H, SCREEN_W, 3], dtype="float32")
        ob_post_ctrl = Input(shape=[SCREEN_H, SCREEN_W, 3], dtype="float32")

        conv1 = Conv2D(32, 8, 8, activation="relu", W_regularizer=WeightRegularizer(l2=self.reg_coef))
        x_pre_ctrl = conv1(ob_pre_ctrl)
        x_post_ctrl = conv1(ob_post_ctrl)
        maxpool1 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))
        x_pre_ctrl = maxpool1(x_pre_ctrl)
        x_post_ctrl = maxpool1(x_post_ctrl)

        conv2 = Conv2D(64, 4, 4, activation="relu", W_regularizer=WeightRegularizer(l2=self.reg_coef))
        x_pre_ctrl = conv2(x_pre_ctrl)
        x_post_ctrl = conv2(x_post_ctrl)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        x_pre_ctrl = maxpool2(x_pre_ctrl)
        x_post_ctrl = maxpool2(x_post_ctrl)

        flatten = Flatten()
        x_pre_ctrl = flatten(x_pre_ctrl)
        x_post_ctrl = flatten(x_post_ctrl)

        x = keras.layers.concatenate([x_pre_ctrl, x_post_ctrl])
        x = Dense(128, activation="relu", W_regularizer=WeightRegularizer(l2=self.reg_coef))(x)
        est_ctrl = Dense(2, W_regularizer=WeightRegularizer(l2=self.reg_coef))(x)

        self.model = Model([ob_pre_ctrl, ob_post_ctrl], [est_ctrl])
        optimizer = Adam(self.init_lr, decay=self.decay)
        self.model.compile(optimizer, loss="mean_squared_error")

        print "Model compiled"
        print [ll.output for ll in self.model.layers()]



model = InverseNet()
model.build_net()