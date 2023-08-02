import cupy as cp
import cv2
import numpy as np


class PredictDepth():
    def __init__(self, img_dim, fov_x, fov_y, max_depth, upscaling: int=1, cam_rot_mat=cp.eye(3)):
        self.fov_x = fov_x  # field of view
        self.fov_y = fov_y  # field of view
        self.max_depth = 1  # max depth camera
        self.depth_scaling = 255 / max_depth
        # focal_length = 0.01
        self.scaling = upscaling
        self.img_dims = img_dim
        self.focal_length_y = cp.tan(fov_y / 360 * cp.pi) / (img_dim[1] / 2)
        self.focal_length_x = cp.tan(fov_x / 360 * cp.pi) / (img_dim[0] / 2)

        self.e_zx = upscaling / cp.tan(fov_x / 360 * cp.pi)
        self.e_zy = upscaling / cp.tan(fov_y / 360 * cp.pi)

        self.nx, self.ny = img_dim[::-1]
        self.img_depth = cp.ones((self.ny, self.nx))
        self.resolution = cp.array([[self.nx * self.scaling], [self.ny * self.scaling]])
        x = cp.arange(self.resolution[0])
        y = cp.arange(self.resolution[1])
        xv, yv = cp.meshgrid(x, y)

        self.image_coord = cp.vstack([xv.flatten() + 0.5 - self.resolution[0] / 2,
                                      yv.flatten() + 0.5 - self.resolution[1] / 2,
                                      cp.ones_like(xv).flatten()])
        self.cam_offset = cp.array([[0], [0], [0]])


        self.local2cam = cam_rot_mat


        self.cam_dot = cp.array([[0], [0], [0.5]])
        self.theta_dot = cp.array([[0.01], [0.03], [0.03]]) * cp.pi

        self.znew = cp.ones_like(self.img_depth)

    def pred_depth(self, img, state_delta):
        img = img/255*self.max_depth
        img_depth = cp.repeat(img, self.scaling, axis=0)
        img_depth = cp.repeat(img_depth, self.scaling, axis=1)

        pos_delta = state_delta[:3].reshape(3,1)
        ori_delta = quat2rot(state_delta[3:])
        # calculate future drone positions (in local drone frame)
        local_dot = pos_delta + ori_delta @ - self.cam_offset
        # calculate future camera positions (in local camera frame)
        cam_dot = self.local2cam @ (local_dot + self.cam_offset)
        self.image_coord[2, :] = cp.array(img_depth.flatten())

        # Calculate positions from depth image in local frame
        local_pos = self.image2local(self.image_coord)
        # transform in future positions in future local frame
        future_pos = self.local2local(local_pos, cam_dot, self.local2cam @ ori_delta.T @ self.local2cam.T)
        # calculate future positions in future camera frame
        future_cam_pos = self.local2image(future_pos)

        # generate future depth image
        abs_cam_pos = cp.round((future_cam_pos + self.resolution / 2 - 0.5)*2, 10)/2
        abs_cam_pos = abs_cam_pos.astype(int)

        ind = (0<= abs_cam_pos[0]) & (abs_cam_pos[0] < self.nx * self.scaling) & \
              (0<= abs_cam_pos[1]) & (abs_cam_pos[1] < self.ny * self.scaling)

        pos_t = abs_cam_pos[:, ind]
        depth_val = future_pos[2, ind]
        future_depth = cp.ones_like(img_depth)*self.max_depth

        final_ind = cp.argsort(-depth_val)
        future_depth[pos_t[1, final_ind], pos_t[0, final_ind]] = depth_val[final_ind]*255/self.max_depth

        z_new = future_depth.get().astype(np.float32)
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

    def local2local(self, point_pos, cam_pos, rot_mat):
        rel_pos = point_pos - cam_pos
        local = rot_mat @ rel_pos
        return local


def quat2rot(q):
    a = q[0]
    b = q[1]
    c = q[2]
    d = q[3]
    rot = cp.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                    [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                    [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]]).T
    return rot