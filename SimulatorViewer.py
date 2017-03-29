import ctypes
from ctypes import byref
from mujoco_py import mjviewer, mjcore, mjconstants, glfw
from mujoco_py.mjlib import mjlib
import numpy as np
import copy
import os
import logging
logger = logging.getLogger(__name__)

mjCAT_ALL = 7

mjPERT_TRANSLATE = 1
mjPERT_ROTATE = 2
mjOBJ_GEOM = 4

SCALE_FORCE = 150
SCALE_TORQUE = 150


class SimulatorViewer(mjviewer.MjViewer):

    def __init__(self, env_path, visible=True, init_width=500, init_height=500, go_fast=False):
        mjviewer.MjViewer.__init__(self, visible, init_width, init_height, go_fast)

        # extra members
        self.simulator = None
        self._env_path = env_path
        self.scale = 0.5
        self.trackbodyid = 1
        self._init_cam = None
        self._perturb = 0
        self._selbody = 0
        self._needselect = 0
        self._selpos = (ctypes.c_double * 3)(0, 0, 0)
        self._refpos = (ctypes.c_double * 3)(0, 0, 0)
        self._refquat = (ctypes.c_double * 4)(0, 0, 0, 0)

    def load_model(self):
        if not os.path.exists(self._env_path):
            raise "Provided XML file does not exist."
        self.set_model(mjcore.MjModel(self._env_path))
        logger.info("Loaded Mujoco model from {}".format(self._env_path))

    def autoscale(self):
        glfw.make_context_current(self.window)
        self.cam.lookat[0] = self.model.stat.center[0]
        self.cam.lookat[1] = self.model.stat.center[1]
        self.cam.lookat[2] = self.model.stat.center[2]
        self.cam.distance = self.scale * self.model.stat.extent
        self.cam.camid = -1
        self.cam.trackbodyid = self.trackbodyid
        width, height = self.get_dimensions()
        mjlib.mjv_updateCameraPose(byref(self.cam), width * 1.0 / height)

        if self._init_cam is None:
            self._init_cam = copy.deepcopy(self.cam)

    def start(self):
        mjviewer.MjViewer.start(self)

        glfw.set_key_callback(self.window, self.handle_keyboard)
        glfw.set_cursor_pos_callback(self.window, self.handle_mouse_move)
        glfw.set_mouse_button_callback(self.window, self.handle_mouse_button)
        glfw.set_scroll_callback(self.window, self.handle_scroll)

    @property
    def selbody(self):
        return self._selbody

    def handle_keyboard(self, window, key, scancode, act, mods):
        # do not act on release
        if act == glfw.RELEASE:
            # self._perturb = 0
            return

        self.gui_lock.acquire()

        if self.simulator is not None:
            if key == glfw.KEY_ENTER:
                self.simulator.record_goal()
            elif key == glfw.KEY_SPACE:
                self.simulator.display_goal()
            elif key == glfw.KEY_H:
                self.simulator.print_help()
            elif key == glfw.KEY_M:
                self.simulator.switch_mode()
            # elif key in [glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT]:
            #     self._perturb = mjPERT_TRANSLATE
            #     axis = 0 if key == glfw.KEY_LEFT or key == glfw.KEY_RIGHT else 1
            #     mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS \
            #                 or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            #     if mod_shift and (key == glfw.KEY_UP or key == glfw.KEY_DOWN):
            #         axis = 2
            #     direct = 1 if key == glfw.KEY_UP or key == glfw.KEY_RIGHT else -1
            #     self._move_object(axis, direct)
            # elif key == glfw.KEY_ESCAPE:
            #     self._perturb = 0
            #     self._selbody = 0


        if mods & glfw.MOD_CONTROL:
            if key == glfw.KEY_A:
                self.cam = copy.deepcopy(self._init_cam)
                self.autoscale()
                self.simulator.evaluate()
            elif key == glfw.KEY_L and self.simulator is not None:
                self.simulator.reload()
            # elif key in [glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT]:
            #     self._perturb = mjPERT_ROTATE
            #     axis = 0 if key == glfw.KEY_LEFT or key == glfw.KEY_RIGHT else 1
            #     mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS \
            #                 or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            #     if mod_shift and (key == glfw.KEY_UP or key == glfw.KEY_DOWN):
            #         axis = 2
            #     direct = 1 if key == glfw.KEY_UP or key == glfw.KEY_RIGHT else -1
            #     self._move_object(axis, direct)

        self.gui_lock.release()
        # logger.info("key event done")

    # def _move_object(self, axis, direct):
    #
    #     # self.gui_lock.acquire()
    #
    #     # perturbation
    #     if self._perturb:
    #         if self._selbody > 0:
    #
    #             refpos_before = np.array(self._refpos)
    #             refquat_before = np.array(self._refquat)
    #
    #             if self._perturb == mjPERT_TRANSLATE:
    #                 displacement = np.zeros(3)
    #                 displacement[axis] = 0.05 * direct
    #                 refpos_after = refpos_before + displacement
    #                 refquat_after = refquat_before
    #                 mjlib.mju_copy3(ctypes.cast(self._refpos, ctypes.POINTER(ctypes.c_double)),
    #                                 refpos_after.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    #             else:
    #                 rot_radian = np.pi / 6
    #                 rot_mat = np.identity(3)
    #                 rot_mat[0, 0] = np.cos(rot_radian)
    #                 rot_mat[1, 1] = np.cos(rot_radian)
    #                 rot_mat[0, 1] = -np.sin(rot_radian)
    #                 rot_mat[1, 0] = np.sin(rot_radian)
    #                 logger.info("rot_mat=\n{}".format(rot_mat))
    #
    #                 rot_quat = (ctypes.c_double * 4) (0, 0, 0, 0)
    #                 mjlib.mju_mat2Quat(rot_quat, rot_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    #                 logger.info("rot_quat={}".format(np.array(rot_quat)))
    #
    #                 # rot_quat = np.zeros(4)
    #                 # rot_quat[3] = 1
    #                 # rot_quat[1] = np.pi / 4.0
    #                 # mjlib.mju_rotVecQuat(ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)),
    #                 #                      ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)),
    #                 #                      ctypes.cast(rot_quat, ctypes.POINTER(ctypes.c_double)))
    #                 refpos_after = refpos_before
    #                 mjlib.mju_copy(ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)),
    #                                ctypes.cast(rot_quat, ctypes.POINTER(ctypes.c_double)), 4)
    #                 refquat_after = np.array(self._refquat)
    #
    #             self._rotated = not np.allclose((refquat_after - refquat_before), 0)
    #             logger.info("self._rotated={}".format(self._rotated))
    #
    #             # debug
    #             logger.info("_refpos before move = {}".format(refpos_before))
    #             logger.info("_refquat before move = {}".format(refquat_before))
    #             logger.info("_refpos after move = {}".format(refpos_after))
    #             logger.info("_refquat after move = {}".format(refquat_after))
    #             logger.info("diff(_refpos) = {}".format(refpos_after - refpos_before))
    #             logger.info("diff(_refquat) = {}".format(refquat_after - refquat_before))
    #             logger.info("-" * 50)

        # self.gui_lock.release()

    def handle_mouse_move(self, window, xpos, ypos):
        # no buttons down: nothing to do
        if not self._button_left_pressed \
                and not self._button_middle_pressed \
                and not self._button_right_pressed:
            return

        # compute mouse displacement, save
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

        # require model
        if not self.model:
            return

        # get current window size
        width, height = glfw.get_framebuffer_size(self.window)

        # get shift key state
        mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS \
                    or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        # determine action based on mouse button
        if self._button_right_pressed:
            action = mjconstants.MOUSE_MOVE_H if mod_shift else mjconstants.MOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mjconstants.MOUSE_ROTATE_H if mod_shift else mjconstants.MOUSE_ROTATE_V
        else:
            action = mjconstants.MOUSE_ZOOM

        self.gui_lock.acquire()

        if self._perturb:
            # perturbation
            if self._selbody > 0:
                # refpos_before = np.array(self._refpos)
                # refquat_before = np.array(self._refquat)
                # logger.info("_refpos before move = {}".format(refpos_before))
                # logger.info("_refquat before move = {}".format(refquat_before))
                mjlib.mjv_moveObject(action, dx, dy, byref(self.cam.pose),
                                    width, height,
                                    ctypes.cast(self._refpos, ctypes.POINTER(ctypes.c_double)),
                                    ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)))
                # refpos_after = np.array(self._refpos)
                # refquat_after = np.array(self._refquat)
                # logger.info("_refpos after move = {}".format(refpos_after))
                # logger.info("_refquat after move = {}".format(refquat_after))
                # logger.info("diff(_refpos) = {}".format(refpos_after - refpos_before))
                # logger.info("diff(_refquat) = {}".format(refquat_after- refquat_before))
                # logger.info("-" * 50)
                # logger.info("(dx, dy)={}".format((dx, dy)))
        # else:
        #     # camera control
        #     mjlib.mjv_moveCamera(action, dx, dy, byref(self.cam), width, height)

        self.gui_lock.release()

    def handle_mouse_button(self, window, button, act, mods):
        # update button state
        self._button_left_pressed = \
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_middle_pressed = \
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self._button_right_pressed = \
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        # update mouse position
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # require model
        if not self.model:
            return

        self.gui_lock.acquire()

        # set perturbation
        newperturb = 0
        if (mods & glfw.MOD_CONTROL) > 0 and (self._selbody > 0):
            # right: translate; left: rotate
            if self._button_right_pressed:
                newperturb = mjPERT_TRANSLATE
            elif self._button_left_pressed:
                newperturb = mjPERT_ROTATE

            # perturbation onset: reset reference
            if newperturb > 0 and not self._perturb != 0:
                id = self._selbody
                mjlib.mju_copy3(ctypes.cast(self._refpos, ctypes.POINTER(ctypes.c_double)),
                                self.model.data.xpos[id].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                mjlib.mju_copy(ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)),
                               self.model.data.xquat[id].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 4)
        self._perturb = newperturb

        # detect double-click (250 msec)
        if act == glfw.PRESS and glfw.get_time() - self._last_click_time < 0.25 and button == self._last_button:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._needselect = 1
            else:
                self._needselect = 2

            # stop perturbation on select
            self._perturb = 0

        # save info
        if act == glfw.PRESS:
            self._last_button = button
            self._last_click_time = glfw.get_time()

        self.gui_lock.release()

    def render(self):
        if not self.data:
            return
        glfw.make_context_current(self.window)

        self.gui_lock.acquire()
        rect = self.get_rect()

        # update simulation statistics
        self.last_render_time = glfw.get_time()

        # create geoms and lights
        mjlib.mjv_makeGeoms(self.model.ptr, self.data.ptr, byref(self.objects), byref(self.vopt), mjCAT_ALL,
                            self._selbody, None, None,
                            # self._selbody,
                            # ctypes.cast(self._refpos, ctypes.POINTER(ctypes.c_double)) if (
                            # self._perturb & mjPERT_TRANSLATE) else None,
                            # ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)) if (
                            # self._perturb & mjPERT_ROTATE) else None,
                            ctypes.cast(self._selpos, ctypes.POINTER(ctypes.c_double)))
        mjlib.mjv_makeLights(self.model.ptr, self.data.ptr, byref(self.objects))

        # update camera
        mjlib.mjv_setCamera(self.model.ptr, self.data.ptr, byref(self.cam))
        mjlib.mjv_updateCameraPose(byref(self.cam), rect.width * 1.0 / rect.height)

        if self._needselect:
            # find selected gemo
            pos = (ctypes.c_double * 3)(0, 0, 0)
            selgeom = mjlib.mjr_select(rect, byref(self.objects),
                                       self._last_mouse_x, rect.height - self._last_mouse_y,
                                       ctypes.cast(pos, ctypes.POINTER(ctypes.c_double)), None,
                                       byref(self.ropt), byref(self.cam.pose), byref(self.con))

            # set lookat point
            if self._needselect == 2:
                if selgeom >= 0:
                    mjlib.mju_copy3(ctypes.cast(self.cam.lookat, ctypes.POINTER(ctypes.c_double)),
                                    ctypes.cast(pos, ctypes.POINTER(ctypes.c_double)))
            else:
                if selgeom >= 0 and self.objects.geoms[selgeom].objtype == mjOBJ_GEOM:
                    # record selection
                    self._selbody = self.model.geom_bodyid[self.objects.geoms[selgeom].objid, 0]

                    if self._selbody < 0 or self._selbody >= self.model.nbody:
                        # clear if invalid
                        self._selbody = 0
                    else:
                        # otherwise compute selpos
                        tmp = (ctypes.c_double * 3)(0, 0, 0)
                        mjlib.mju_sub3(ctypes.cast(tmp, ctypes.POINTER(ctypes.c_double)),
                                       ctypes.cast(pos, ctypes.POINTER(ctypes.c_double)),
                                       self.data.xpos[self._selbody].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                        assert self.model.data.xmat.shape[1] == 9
                        mjlib.mju_mulMatTVec(ctypes.cast(self._selpos, ctypes.POINTER(ctypes.c_double)),
                                             self.data.xmat[self._selbody].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             ctypes.cast(tmp, ctypes.POINTER(ctypes.c_double)), 3, 3)
                else:
                    self._selbody = 0
            self._needselect = 0

        # render rbg
        mjlib.mjr_render(0, rect, byref(self.objects), byref(self.ropt), byref(self.cam.pose), byref(self.con))

        self.gui_lock.release()

    def advance(self):

        # self.gui_lock.acquire()

        # perturbations
        if self._selbody > 0:

            if self.model.body_jntnum[self._selbody, 0] == 0 and self.model.body_parentid[self._selbody, 0] == 0:
                # fixed object: edit
                mjlib.mjv_mouseEdit(self.model.ptr, self.data.ptr,
                                    self._selbody, self._perturb,
                                    ctypes.cast(self._refpos, ctypes.POINTER(ctypes.c_double)),
                                    ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)))
            else:
                # # reset reference frame
                # if self._perturb == 0:
                #     id = self._selbody
                #     mjlib.mju_copy3(ctypes.cast(self._refpos, ctypes.POINTER(ctypes.c_double)),
                #                     self.model.data.xpos[id].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                #     mjlib.mju_copy(ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)),
                #                    self.model.data.xquat[id].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 4)
                # else:
                # movable object: set mouse perturbation
                tmp = np.zeros_like(self.data.xfrc_applied)
                mjlib.mjv_mousePerturb(self.model.ptr, self.data.ptr,
                                       self._selbody, self._perturb,
                                       ctypes.cast(self._refpos, ctypes.POINTER(ctypes.c_double)),
                                       ctypes.cast(self._refquat, ctypes.POINTER(ctypes.c_double)),
                                       tmp[self._selbody].ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                # logger.info("perturb={}".format("TRANS" if self._perturb==mjPERT_TRANSLATE else "ROTATE"))
                # logger.info("tmp={}".format(tmp[self._selbody]))
                self.data.xfrc_applied = tmp

        # clear perturbation
        if self._selbody > 0:
            mjlib.mju_zero(self.data.xfrc_applied[self._selbody].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 6)

        # advance simulation
        # self.model.step()

        # self.gui_lock.release()

    def apply_force_torque(self, selbody, xfrc):

        if selbody > 0 and xfrc is not None:
            tmp = np.copy(self.data.xfrc_applied)
            tmp[selbody] = xfrc
            tmp[selbody, :3] *= SCALE_FORCE
            tmp[selbody, 3:] *= SCALE_TORQUE
            self.data.xfrc_applied = tmp

        self.model.step()

        if selbody > 0 and xfrc is not None:
            mjlib.mju_zero(self.data.xfrc_applied[selbody].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 6)

    def get_glfw_time(self):
        return glfw.get_time()