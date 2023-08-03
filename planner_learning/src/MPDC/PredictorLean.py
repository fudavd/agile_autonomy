import cv2
import numpy as np

class PredictDepth():
    def __init__(self,):
        self.fov_x = 80  # field of view
        self.fov_y = 80  # field of view
        self.depth_scaling = 255
        # focal_length = 0.01
        self.scaling = 2
        self.img_dims = (224, 224)

        self.e_zx = self.scaling / np.tan(self.fov_x / 360 * np.pi)
        self.e_zy = self.scaling / np.tan(self.fov_y / 360 * np.pi)

        self.nx, self.ny = self.img_dims[::-1]
        self.img_depth = np.ones((self.ny, self.nx))
        self.resolution = np.array([[self.nx * self.scaling], [self.ny * self.scaling]])
        x = np.arange(self.resolution[0])
        y = np.arange(self.resolution[1])
        xv, yv = np.meshgrid(x, y)

        self.image_coord = np.vstack([xv.flatten() + 0.5 - self.resolution[0] / 2,
                                      yv.flatten() + 0.5 - self.resolution[1] / 2,
                                      np.ones_like(xv).flatten()])

        cam_reori = np.eye(3)
        # cam_reori = np.array([[1, 0, 0],
        #                       [0, np.sqrt(3) / 2, 0.5],
        #                       [0, -0.5, np.sqrt(3) / 2]])

        self.local2cam = cam_reori @ np.array([[-1, 0, 0],
                                               [0, 0, -1],
                                               [0, -1, 0]])

        self.znew = np.ones_like(self.img_depth)

    def pred_depth(self, img, pos_delta):
        img = img/255
        img_depth = np.repeat(img, self.scaling, axis=0)
        img_depth = np.repeat(img_depth, self.scaling, axis=1)

        # calculate future drone positions (in local drone frame)
        local_dot = pos_delta
        # calculate future camera positions (in local camera frame)
        cam_dot = self.local2cam @ local_dot
        self.image_coord[2, :] = np.array(img_depth.flatten())

        # Calculate positions from depth image in local frame
        local_pos = self.image2local(self.image_coord)
        # transform in future positions in future local frame
        future_pos = self.local2local(local_pos, cam_dot)
        # calculate future positions in future camera frame
        future_cam_pos = self.local2image(future_pos)

        # generate future depth image
        abs_cam_pos = np.round((future_cam_pos + self.resolution / 2 - 0.5)*2, 10)/2
        abs_cam_pos = abs_cam_pos.astype(int)

        ind = (0<= abs_cam_pos[0]) & (abs_cam_pos[0] < self.nx * self.scaling) & \
              (0<= abs_cam_pos[1]) & (abs_cam_pos[1] < self.ny * self.scaling)

        pos_t = abs_cam_pos[:, ind]
        depth_val = future_pos[2, ind]
        future_depth = np.ones_like(img_depth, dtype=np.float32)

        final_ind = np.argsort(-depth_val)
        future_depth[pos_t[1, final_ind], pos_t[0, final_ind]] = depth_val[final_ind]*255.0

        z_new = future_depth
        z_new = cv2.medianBlur(z_new, 3)
        z_new = cv2.resize(z_new, (self.nx, self.ny), cv2.INTER_NEAREST)
        z_new = cv2.medianBlur(z_new, 3)
        return z_new

    def image2local(self, image_pos):
        local_x = (image_pos[0]) / self.e_zx * image_pos[2]
        local_y = -(image_pos[1]) / self.e_zy * image_pos[2]
        return np.vstack([local_x, local_y, image_pos[2]])

    def local2image(self, local_pos):
        cam_x = self.e_zx / local_pos[2] * local_pos[0]
        cam_y = -self.e_zy / local_pos[2] * local_pos[1]
        return np.vstack([cam_x, cam_y])

    def local2local(self, point_pos, cam_pos):
        rel_pos = point_pos - cam_pos
        return rel_pos


import cv2
import cupy as cp


class PredictDepth_cp():
    def __init__(self,):
        self.fov_x = 80  # field of view
        self.fov_y = 80  # field of view
        self.depth_scaling = 255
        # focal_length = 0.01
        self.scaling = 2
        self.img_dims = (224, 224)

        self.e_zx = self.scaling / cp.tan(self.fov_x / 360 * cp.pi)
        self.e_zy = self.scaling / cp.tan(self.fov_y / 360 * cp.pi)

        self.nx, self.ny = self.img_dims[::-1]
        self.img_depth = cp.ones((self.ny, self.nx))
        self.resolution = cp.array([[self.nx * self.scaling], [self.ny * self.scaling]])
        x = cp.arange(self.resolution[0])
        y = cp.arange(self.resolution[1])
        xv, yv = cp.meshgrid(x, y)

        self.image_coord = cp.vstack([xv.flatten() + 0.5 - self.resolution[0] / 2,
                                      yv.flatten() + 0.5 - self.resolution[1] / 2,
                                      cp.ones_like(xv).flatten()])

        # cam_reori = cp.eye(3)
        cam_reori = cp.array([[1, 0, 0],
                              [0, np.sqrt(3) / 2, 0.5],
                              [0, -0.5, np.sqrt(3) / 2]])

        self.local2cam = cam_reori @ cp.array([[-1, 0, 0],
                                               [0, 0, -1],
                                               [0, -1, 0]])

        self.znew = cp.ones_like(self.img_depth)

    def pred_depth(self, img, pos_delta):
        img = img/255
        img_depth = cp.repeat(img, self.scaling, axis=0)
        img_depth = cp.repeat(img_depth, self.scaling, axis=1)

        # calculate future drone positions (in local drone frame)
        local_dot = cp.array(pos_delta)
        # calculate future camera positions (in local camera frame)
        cam_dot = self.local2cam @ local_dot
        self.image_coord[2, :] = cp.array(img_depth.flatten())

        # Calculate positions from depth image in local frame
        local_pos = self.image2local(self.image_coord)
        # transform in future positions in future local frame
        future_pos = self.local2local(local_pos, cam_dot)
        # calculate future positions in future camera frame
        future_cam_pos = self.local2image(future_pos)

        # generate future depth image
        abs_cam_pos = cp.round((future_cam_pos + self.resolution / 2 - 0.5)*2, 10)/2
        abs_cam_pos = abs_cam_pos.astype(int)

        ind = (0<= abs_cam_pos[0]) & (abs_cam_pos[0] < self.nx * self.scaling) & \
              (0<= abs_cam_pos[1]) & (abs_cam_pos[1] < self.ny * self.scaling)

        pos_t = abs_cam_pos[:, ind]
        depth_val = future_pos[2, ind]
        future_depth = cp.ones_like(img_depth, dtype=cp.float32)

        final_ind = cp.argsort(-depth_val)
        future_depth[pos_t[1, final_ind], pos_t[0, final_ind]] = depth_val[final_ind]*255.0

        z_new = future_depth.get()
        z_new = cv2.medianBlur(z_new, 3)
        z_new = cv2.resize(z_new, (self.nx, self.ny), cv2.INTER_NEAREST)
        z_new = cv2.medianBlur(z_new, 3)
        return z_new

    def image2local(self, image_pos):
        local_x = (image_pos[0]) / self.e_zx * image_pos[2]
        local_y = -(image_pos[1]) / self.e_zy * image_pos[2]
        return cp.vstack([local_x, local_y, image_pos[2]])

    def local2image(self, local_pos):
        cam_x = self.e_zx / local_pos[2] * local_pos[0]
        cam_y = -self.e_zy / local_pos[2] * local_pos[1]
        return cp.vstack([cam_x, cam_y])

    def local2local(self, point_pos, cam_pos):
        rel_pos = point_pos - cam_pos
        return rel_pos
