import tensorflow as tf
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)


DECAY_STEPS = 10000
DECAY_RATE = 0.9


class TNet:

    def __init__(self, sess, num_input_ch, input_h, input_w, init_lr):
        self._sess = sess
        self._num_input_ch = num_input_ch
        self._input_h = input_h
        self._input_w = input_w

        self._theta_inverse = {}
        self._theta_forward = {}
        self._saver = None

        self._init_lr = init_lr
        self._lambda = 0.0

        self._ctrl_dim_names = ["x", "y", "z", "roll", "yaw", "pitch"]

    def build_inverse_model(self):
        """
        Given images of pre and post control application, estimate applied control
        :return:
        """
        # conv1
        a1_pre_ctrl = tf.layers.conv2d(inputs=self.ob_pre_ctrl,
                                       filters=32,
                                       kernel_size=(8, 8),
                                       strides=(4, 4),
                                       activation=tf.nn.relu,
                                       name="conv1")
        a1_post_ctrl = tf.layers.conv2d(inputs=self.ob_post_ctrl,
                                        filters=32,
                                        kernel_size=(8, 8),
                                        strides=(4, 4),
                                        activation=tf.nn.relu,
                                        name="conv1",
                                        reuse=True)
        tf.add_to_collection(name="activation", value=a1_pre_ctrl)
        tf.add_to_collection(name="activation", value=a1_post_ctrl)

        # conv2
        a2_pre_ctrl = tf.layers.conv2d(inputs=a1_pre_ctrl,
                                       filters=64,
                                       kernel_size=(4, 4),
                                       strides=(2, 2),
                                       activation=tf.nn.relu,
                                       name="conv2")
        a2_post_ctrl = tf.layers.conv2d(inputs=a1_post_ctrl,
                                        filters=64,
                                        kernel_size=(4, 4),
                                        strides=(2, 2),
                                        activation=tf.nn.relu,
                                        name="conv2",
                                        reuse=True)
        tf.add_to_collection(name="activation", value=a2_pre_ctrl)
        tf.add_to_collection(name="activation", value=a2_post_ctrl)
        self.fea_pre_ctrl = a2_pre_ctrl
        self.fea_post_ctrl = a2_post_ctrl

        # relu1
        # input_size = 9 * 9 * 64
        # a3_pre_ctrl = tf.layers.dense(inputs=tf.reshape(a2_pre_ctrl, shape=[-1, input_size]),
        #                               units=512,
        #                               activation=tf.nn.relu,
        #                               use_bias=False,
        #                               name="fc1")
        # a3_post_ctrl = tf.layers.dense(inputs=tf.reshape(a2_post_ctrl, shape=[-1, input_size]),
        #                                units=512,
        #                                activation=tf.nn.relu,
        #                                use_bias=False,
        #                                name="fc1",
        #                                reuse=True)
        # tf.add_to_collection(name="activation", value=a3_pre_ctrl)
        # tf.add_to_collection(name="activation", value=a3_post_ctrl)
        #
        # a3_concat = tf.concat(values=[a3_pre_ctrl, a3_post_ctrl],
        #                       axis=1,
        #                       name="fc1_concat")

        logger.info("a2_pre_ctrl.shape={}".format(a2_pre_ctrl.shape))
        input_size = np.prod(a2_pre_ctrl.get_shape().as_list()[1:])
        assert input_size == 9 * 9 * 64, "input_size={}, expected={}".format(input_size, 81*64)
#        input_size = 9 * 9 * 64
        a2_pre_ctrl_flat = tf.reshape(a2_pre_ctrl, shape=[-1, input_size])
        a2_post_ctrl_flat = tf.reshape(a2_post_ctrl, shape=[-1, input_size])
        a2_flat = tf.concat([a2_pre_ctrl_flat,a2_post_ctrl_flat], axis=1)

        a3 = tf.layers.dense(inputs=a2_flat,
                             units=128,
                             activation=tf.nn.relu,
                             name="fc1")
        tf.add_to_collection(name="activation", value=a3)

        # a4 = tf.layers.dense(inputs=a3,
        #                      units=1024,
        #                      activation=tf.nn.relu,
        #                      use_bias=False,
        #                      name="fc2")
        # tf.add_to_collection(name="activation", value=a4)

        self.est_ctrl = tf.layers.dense(inputs=a3,
                                        units=6,
                                        activation=tf.nn.tanh,
                                        name="est_ctrl")
        tf.add_to_collection(name="activation", value=self.est_ctrl)

        # ctrl_fc2 = []
        # ctrl_fc3 = []
        # est_ctrl_by_dim = []
        # for ctrl_dim in self._ctrl_dim_names:
        #     with tf.variable_scope("subnet_" + ctrl_dim):
        #         fc2_dim = tf.layers.dense(inputs=a2_flat,
        #                                   units=1024,
        #                                   activation=tf.nn.relu,
        #                                   use_bias=False,
        #                                   name="fc1_" + ctrl_dim)
        #         fc3_dim = tf.layers.dense(inputs=fc2_dim,
        #                                   units=1024,
        #                                   activation=tf.nn.relu,
        #                                   use_bias=False,
        #                                   name="fc2_" + ctrl_dim)
        #         est_ctrl_dim = tf.layers.dense(inputs=fc3_dim,
        #                                        units=1,
        #                                        activation=tf.nn.tanh,
        #                                        use_bias=False,
        #                                        name="est_ctrl_" + ctrl_dim)
        #         tf.add_to_collection(name="activation", value=fc2_dim)
        #         tf.add_to_collection(name="activation", value=fc3_dim)
        #         tf.add_to_collection(name="activation", value=est_ctrl_dim)
        #         # ctrl_fc2.append(fc2_dim)
        #         # ctrl_fc3.append(fc3_dim)
        #         est_ctrl_by_dim.append(est_ctrl_dim)

        # output
        # self.est_ctrl = tf.concat(est_ctrl_by_dim, axis=1, name="est_ctrl")

        # return est_ctrl_by_dim

    def build_forward_model(self):
        """
        Given current image and the control to apply,
        estimate the feature representation of the image after control application
        :return:
        """
        with tf.variable_scope("reshape"):
            tmp = tf.unstack(self.fea_pre_ctrl, axis=-1)
            self.fea_pre_ctrl_sliced = [tf.reshape(t, shape=[-1, 7*7]) for t in tmp]

        # relu1
        a1 = []
        for fea_pre_ctrl in self.fea_pre_ctrl_sliced:
            a1_tmp = tf.layers.dense(inputs=fea_pre_ctrl,
                                     units=512,
                                     activation=tf.nn.relu,
                                     name="fc1",
                                     reuse=True if len(a1) > 0 else None)
            a1.append(a1_tmp)

        # softmax1
        a2 = tf.layers.dense(inputs=tf.concat(a1, 1),
                             units=64,
                             name="fc2")
        self.att_alpha = tf.nn.softmax(logits=a2,
                                       name="att_alpha")
        tf.summary.histogram(name="attention",
                             values=self.att_alpha)

        # soft attention
        with tf.variable_scope("weighted_sum"):
            tmp = tf.multiply(self.fea_pre_ctrl, tf.reshape(self.att_alpha, shape=[-1, 1, 1, 64]))
            z = tf.reduce_sum(tmp, axis=-1, name="z_vector")
            z_flatten = tf.reshape(z, shape=[-1, 7*7], name="z_vector_flatten")

        # relu2
        a3 = tf.layers.dense(inputs=tf.concat([z_flatten, self.ctrl], 1),
                             units=512,
                             activation=tf.nn.relu,
                             name="fc3")
        tf.summary.histogram(name="a3",
                             values=a3)

        # output
        self.est_x_post_ctrl = tf.layers.dense(inputs=a3,
                                               units=256,
                                               activation=tf.nn.tanh,
                                               name="est_x_post_ctrl")


    def build_net(self):

        with tf.variable_scope("inputs"):
            self.ob_pre_ctrl = tf.placeholder(dtype="float32",
                                              shape=[None, self._input_h, self._input_w, self._num_input_ch],
                                              name="ob_pre_ctrl")
            self.ob_post_ctrl = tf.placeholder(dtype="float32",
                                               shape=[None, self._input_h, self._input_w, self._num_input_ch],
                                               name="ob_post_ctrl")
            self.ctrl = tf.placeholder(dtype="float32",
                                       shape=[None, 6],
                                       name="ctrl")
            # ctrl_splitted = tf.unstack(self.ctrl, axis=-1, name="ctrl_splitted")

            # add image summary
            ob_pre_ctrl_split = tf.unstack(self.ob_pre_ctrl, axis=-1)
            ob_post_ctrl_split = tf.unstack(self.ob_post_ctrl, axis=-1)
            for i, (pre_img, post_img) in enumerate(zip(ob_pre_ctrl_split, ob_post_ctrl_split)):
                tf.summary.image(name="0_pre_ctrl_" + str(i),
                                 tensor=tf.expand_dims(pre_img, axis=-1),
                                 max_outputs=1)
                tf.summary.image(name="1_post_ctrl_" + str(i),
                                 tensor=tf.expand_dims(post_img, axis=-1),
                                 max_outputs=1)

        with tf.variable_scope("inverse_model"):
            self.build_inverse_model()

        # logger.info("Building forward model ...")
        # with tf.variable_scope("forward_model"):
        #     self.build_forward_model()

        with tf.variable_scope("losses"):
            # inverse_losses = []
            # for idx, (est_ctrl, ctrl) in enumerate(zip(est_ctrl_by_dim, ctrl_splitted)):
            #     loss_name = self._ctrl_dim_names[idx] + "_loss"
            #     with tf.variable_scope(loss_name):
            #         loss_dim = tf.reduce_mean(tf.abs(est_ctrl - ctrl), name=loss_name)
            #         tf.summary.scalar(name=loss_name, tensor=loss_dim)
            #     inverse_losses.append(loss_dim)
            # self.inverse_loss = tf.add_n(inputs=inverse_losses, name="inverse_loss")
            # self.inverse_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.est_ctrl - self.ctrl), axis=1),
            #                                    name="L1_inverse_loss")
            self.inverse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.est_ctrl - self.ctrl), axis=1),
                                               name="inverse_loss")
            self.loss = self.inverse_loss

            # add summaries
            tf.summary.scalar("inverse_loss", self.inverse_loss)
            tf.summary.scalar("total_loss", self.loss)

        with tf.variable_scope("optimization"):
            self.global_step = tf.Variable(0, trainable=False)
            self.lr = tf.train.exponential_decay(learning_rate=self._init_lr,
                                                 global_step=self.global_step,
                                                 decay_steps=DECAY_STEPS,
                                                 decay_rate=DECAY_RATE,
                                                 staircase=True)
            tf.summary.scalar("learning_rate", self.lr)
            # optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=self.lr)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads = optimizer.compute_gradients(loss=self.loss, var_list=tf.trainable_variables())
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        with tf.variable_scope("activations"):
            for a in tf.get_collection("activation"):
                a_name = a.name.replace(":", "_")
                tf.summary.histogram(name=a_name, values=a)
        with tf.variable_scope("weights"):
            for v in tf.trainable_variables():
                v_name = v.name.replace(":", "_")
                tf.summary.histogram(name=v_name, values=v)
        with tf.variable_scope("gradients"):
            for g in grads:
                g_name = g[1].name.replace(":", "_")
                tf.summary.histogram(name=g_name, values=g[0])

        logger.info("Network built.")


    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver

    def save_net(self, model_dir, step=None):
        logger.info("Saving checkpoint to {} ...".format(model_dir))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.saver.save(self._sess, model_dir + "/pre_trained", global_step=step)
        logger.info("Checkpoint saved.")

    def load_net(self, model_dir):
        logger.info("Loading network from {}".format(model_dir))
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(model_dir, ckpt_name)
            self.saver.restore(self._sess, fname)
            logger.info("Network loaded!")
            return True
        else:
            logger.info("Failed to load checkpoint from {}.".format(model_dir))
            return False
