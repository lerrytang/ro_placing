import tensorflow as tf
from TNet import TNet
import numpy as np
import time
import os
from Simulator import SCREEN_H, SCREEN_W
import logging
logger = logging.getLogger(__name__)


DB_CAPACITY = 100000
DB_SAMPLE_PROB = 0.0


class Agent:

    def __init__(self, simulator, log_dir=None):
        self._simulator = simulator

        self._log_dir = log_dir

        self._model = None
        self._model_loaded = False

    def pre_train(self, max_iter, batch_size=64, init_lr=0.01, data_file=None, model_file=None):

        logger.info("SCREEN_H={}, SCREEN_W={}, batch_size={}, init_lr={}".format(SCREEN_H, SCREEN_W, batch_size, init_lr))
        if data_file is not None:
            logger.info("data file specified: {}".format(data_file))
        if model_file is not None:
            logger.info("model file specified: {}".format(model_file))

        current_time = time.strftime("%Y%m%d%H%M%S")
        folder_name = self._simulator.env_name + "_" + current_time
        log_folder = os.path.join(self._log_dir, folder_name)
        pre_train_data_file = os.path.join(log_folder, "pre_train_data.npz")
        if not os.path.exists(log_folder):
            os.mkdir(log_folder)
        model_folder = os.path.join(log_folder, "model")
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)

        num_obj = self._simulator.get_num_obj()

        if data_file is not None and os.path.exists(data_file):
            # load data base
            npzdata = np.load(data_file)
            db_pre_ctrl = npzdata["db_pre_ctrl"]
            db_post_ctrl = npzdata["db_post_ctrl"]
            db_ctrl = npzdata["db_ctrl"]
            db_size = db_ctrl.shape[0]
            logger.info("Database loaded from {}".format(data_file))
        else:
            # create data base for pre-training data
            db_pre_ctrl = np.zeros([DB_CAPACITY, self._simulator.frame_skip, SCREEN_H, SCREEN_W], dtype="uint8")
            db_post_ctrl = np.zeros([DB_CAPACITY, self._simulator.frame_skip, SCREEN_H, SCREEN_W], dtype="uint8")
            db_ctrl = np.zeros([DB_CAPACITY, 6], dtype="float32")
            db_size = 0
            logger.info("Populating database ...")
            while db_size < DB_CAPACITY:
                if db_size % 1000 == 0:
                    logger.info("DB size: {}/{}".format(db_size, DB_CAPACITY))
                ob_pre_ctrl, ctrl, ob_post_ctrl = self._rand_act_in_sim(num_obj)
                db_pre_ctrl[db_size] = ob_pre_ctrl
                db_post_ctrl[db_size] = ob_post_ctrl
                db_ctrl[db_size] = ctrl
                db_size += 1
            np.savez(pre_train_data_file,
                     db_pre_ctrl=db_post_ctrl,
                     db_ctrl=db_ctrl,
                     db_post_ctrl=db_post_ctrl)

        with tf.Session() as sess:

            self._model = TNet(sess, self._simulator.frame_skip, SCREEN_H, SCREEN_W, init_lr)
            self._model.build_net()

            if model_file is not None and os.path.exists(model_file):
                res = self._model.load_net(model_file)
                logger.info("Model load: {}".format("Success" if res else "Failed"))
            else:
                init = tf.global_variables_initializer()
                sess.run(init)

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(log_folder, sess.graph)


            # training
            logger.info("Starting pre-training for {} iterations ...".format(max_iter))
            n_iter = 0
            db_offset = 0
            while n_iter <= max_iter:

                # train (inverse model only)
                rand_db_idx = np.random.choice(DB_CAPACITY, size=batch_size, replace=False)
                ob_pre_ctrl_buffer = self._simulator.transform_imgs(db_pre_ctrl[rand_db_idx])
                ob_post_ctrl_buffer = self._simulator.transform_imgs(db_post_ctrl[rand_db_idx])
                ctrl_buffer = db_ctrl[rand_db_idx]
                loss, summary, est_ctrl, lr, n_iter, _ = sess.run([self._model.loss,
                                                           merged,
                                                           self._model.est_ctrl,
                                                           self._model.lr,
                                                           self._model.global_step,
                                                           self._model.train_op],
                                                          feed_dict={
                                                              self._model.ctrl: ctrl_buffer,
                                                              self._model.ob_pre_ctrl: ob_pre_ctrl_buffer,
                                                              self._model.ob_post_ctrl: ob_post_ctrl_buffer
                                                          })

                # write statistics
                if (n_iter-1) % 100 == 0:
                    logger.info("Progress={0}/{1}\tloss={2:.4f}\tlr={3:.6f}\tdb_size={4}".format(n_iter-1,
                                                                                                max_iter,
                                                                                                loss,
                                                                                                lr,
                                                                                                db_size))
                    logger.info("diff_ctrl:{}".format(np.mean(est_ctrl - ctrl_buffer, axis=0)))
                    writer.add_summary(summary, global_step=n_iter - 1)

                # sample from simulator
                if np.random.rand() < DB_SAMPLE_PROB:
                    logger.info("Sampling from simulator ...")
                    sample_count = 0
                    while sample_count < batch_size:
                        ob_pre_ctrl, ctrl, ob_post_ctrl = self._rand_act_in_sim(num_obj)
                        db_pre_ctrl[db_offset] = ob_pre_ctrl
                        db_post_ctrl[db_offset] = ob_post_ctrl
                        db_ctrl[db_offset] = ctrl
                        db_offset = (db_offset + 1) % db_size
                        sample_count += 1

                if n_iter % 5000 == 0:
                    self._model.save_net(model_folder)
                    np.savez(pre_train_data_file,
                             db_pre_ctrl=db_post_ctrl,
                             db_ctrl=db_ctrl,
                             db_post_ctrl=db_post_ctrl)

            # save the pre-trained net
            self._model.save_net(model_folder)

    def _rand_act_in_sim(self, num_obj):
        ob_pre_ctrl = self._simulator.perturb_obj(0, None)
        selobj = np.random.randint(low=0, high=num_obj) + 1
        ctrl = np.random.rand(6) * 2 - 1
        ob_post_ctrl = self._simulator.perturb_obj(selobj, ctrl)
        return ob_pre_ctrl, ctrl, ob_post_ctrl

