import numpy as np
import six
import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from SimulatorViewer import SimulatorViewer
import logging
logger = logging.getLogger(__name__)


SCREEN_H = 84
SCREEN_W = 84


class Simulator(object):

    def __init__(self, win_w, win_h, frame_skip, env_path, img_dir):
        self._win_w = win_w
        self._win_h = win_h
        self.frame_skip = frame_skip
        self._env_path = env_path
        self.env_name = env_path.split("/")[-1].split(".")[0]
        self._img_dir = img_dir

        self._viewer = None

        self._goal_xpos = None
        self._goal_xquat = None
        self._goal_img = None

        self.observation = np.zeros([self.frame_skip, SCREEN_H, SCREEN_W], dtype="uint8")
        logger.info("observation initialized. shape={}".format(self.observation.shape))

    @property
    def viewer(self):
        if self._viewer is None:
            self._viewer = SimulatorViewer(self._env_path, init_width=self._win_w, init_height=self._win_h)
            self._viewer.scale = 0.65
            self._viewer.trackbodyid = -1
            self._viewer.simulator = self
            self._viewer.start()
            self._viewer.load_model()
        return self._viewer

    def reload(self):
        if self._viewer is not None:
            self._viewer.finish()
            self._viewer = None

    def record_goal(self):
        if self.viewer.selbody>0:
            logger.info("Goal state NOT captured. De-select object first.")
            return

        objs_xpos, objs_xquat = self.get_obj_pos_quat()

        # record position and orientation of each object
        if self._goal_xpos is None or self._goal_xquat is None:
            self._goal_xpos = {}
            self._goal_xquat = {}
        obj_names = np.sort(objs_xpos.keys())
        for obj_name in obj_names:
            logger.info("Target object name: {}".format(obj_name))
            if obj_name in self._goal_xpos.keys():
                logger.info("Diff(xpos)={}".format(objs_xpos[obj_name] - self._goal_xpos[obj_name]))
                logger.info("Diff(xquat)={}".format(objs_xquat[obj_name] - self._goal_xquat[obj_name]))
            self._goal_xpos[obj_name] = objs_xpos[obj_name]
            self._goal_xquat[obj_name] = objs_xquat[obj_name]
            logger.info("Current(xpos)={}".format(self._goal_xpos[obj_name]))
            logger.info("Current(xquat)={}".format(self._goal_xquat[obj_name]))

        # record image
        self._goal_img = self.get_current_scene()
        self.display_goal()

    def get_num_obj(self):
        logger.info("#objects in the simulator: {}".format(self.viewer.model.njnt))
        return self.viewer.model.njnt

    def get_current_scene(self):
        # self.viewer.render()
        data, width, height = self.viewer.get_image()
        return np.fromstring(data,dtype="uint8").reshape(height, width, 3)[::-1,:,:]

    def get_obj_pos_quat(self):
        objs_xpos = {}
        objs_xquat = {}
        n_obj = self.get_num_obj()
        for i in xrange(n_obj):
            obj_name = "obj" + str(i + 1)
            idx = self.viewer.model.body_names.index(six.b(obj_name))
            objs_xpos[obj_name] = self.viewer.model.data.xpos[idx]
            objs_xquat[obj_name] = self.viewer.model.data.xquat[idx]
        return objs_xpos, objs_xquat

    def display_goal(self):
        if self._goal_img is not None:
            plt.imshow(self._goal_img)
            plt.axis("off")
            plt.title("Goal image")
            plt.show()

    def evaluate(self, alpha=1.0, beta=1.0, gamma=1.0):
        if self._goal_img is None or self._goal_xpos is None or self._goal_xquat is None:
            logger.info("Goal state unavailable.")
            return
        current_scene = self.get_current_scene()
        current_xpos, current_xquant = self.get_obj_pos_quat()
        # TODO: all losses are L1 norm currently, define more appropriate ones
        loss_image = np.sum(np.abs(current_scene - self._goal_img))
        loss_xpos = 0
        loss_xquant = 0
        obj_names = np.sort(current_xpos.keys())
        for obj_name in obj_names:
            loss_xpos += np.sum(np.abs(current_xpos[obj_name]-self._goal_xpos[obj_name]))
            loss_xquant += np.sum(np.abs(current_xquant[obj_name]-self._goal_xquat[obj_name]))
        total_loss = alpha*loss_image + beta*loss_xpos + gamma*loss_xquant
        logger.info("loss(image)={0:.4f}\tloss(xpos)={1:.4f}\t"
                    "loss(xquat)={2:.4f}\tloss(total)={0:.4f}".format(loss_image, loss_xpos, loss_xquant, total_loss))
        return current_scene, total_loss

    def print_help(self):
        logger.info("--- Help on object manipulation ---")
        logger.info("Double left click to select object.")
        logger.info("'Ctrl' + left mouse movement to rotate the selected object.")
        logger.info("'Ctrl' + right mouse movement to translate the selected object.")
        logger.info("'Enter' to take a snapshot of a desired goal state.")
        logger.info("'Space' to show the most recently taken goal state.")
        logger.info("'Ctrl' + 'L' to reset the environment.")
        logger.info("'Ctrl' + 'A' to evaluate (reset camera pose too).")
        logger.info("'H' or 'h' to show this help message.")
        logger.info("------------------------------------")

    def run_sim(self):
        logger.info("Starting simulation ...")
        self.print_help()
        while not self.viewer.should_stop():
            # simulate
            self.viewer.advance()
            self.viewer.model.step()
            # display
            self.viewer.loop_once()
        logger.info("Goodbye")

    def perturb_obj(self, selbody, xfrc):

        for _ in xrange(self.frame_skip):
            start_sim_time = self.viewer.data.time
            while self.viewer.data.time - start_sim_time < 0.01:
                self.viewer.apply_force_torque(selbody, xfrc)

            self.viewer.loop_once()
            ob = np.copy(self.get_current_scene())
            ob = cv2.resize(cv2.cvtColor(ob, cv2.COLOR_RGB2GRAY), (SCREEN_W, SCREEN_H))
            self.observation[:self.frame_skip-1] = self.observation[1:]
            self.observation[-1] = ob

        return np.copy(self.observation)

    def transform_imgs(self, imgs):
        """
        Normalize pixel values along each channel
        """
        res = imgs.astype("float32") / 255.0
        res = np.transpose(res, [0, 2, 3, 1])
        for id, img in enumerate(res):
            img[:, :, 1:] = np.diff(img, axis=-1)
            img -= np.mean(img.reshape([SCREEN_W * SCREEN_H, -1]), axis=0, keepdims=True)
            res[id] = img
        return res
