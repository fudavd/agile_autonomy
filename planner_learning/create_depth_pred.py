#!/usr/bin/env python3

import argparse
import os.path
import sys
sys.path.append("./src/PlannerLearning/models")
import cupy as cp
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
# from src.PlannerLearning.models.plan_learner import PlanLearner
from src.PlannerLearning.models.data_loader import create_dataset
from src.MPDC.Predictor import PredictDepth
from config.settings import create_settings



def main():
    parser = argparse.ArgumentParser(description='Train Planning Network')
    parser.add_argument('--settings_file',
                        help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='train')

    ny = settings.img_height
    nx = settings.img_width
    fov_y = 80
    fov_x = 80

    cam_reori = cp.array([[1, 0, 0],
                          [0, np.sqrt(3)/2, 0.5],
                          [0, -0.5, np.sqrt(3)/2]])

    cam_rot = cam_reori @ cp.array([[-1,  0,  0],
                                   [ 0,  0, -1],
                                   [ 0, -1,  0]])

    predictor = PredictDepth((ny,nx), fov_x, fov_y, 10, 2, cam_rot)

    dataset_val = create_dataset(settings.val_dir,
                                 settings, training=False)
    dataset_train = create_dataset(settings.train_dir,
                                   settings, training=False)
    datasets = [dataset_train, dataset_val]
    for dataset in datasets:
        print(f"Append dataset {dataset.directory} with depth prediction")
        new_indices = np.where(np.diff(dataset.traj_idx_num) < 1)[0] + 1
        new_indices = np.hstack((new_indices, len(dataset.traj_idx_num)))

        ind_min = 0
        for ind_max in new_indices:

            samples = np.arange(ind_min, ind_max)
            for sample in samples:
                features, label, _ = dataset._dataset_map(sample)
                file_name = dataset.depth_filenames[sample]
                dir_name = os.path.dirname(file_name)
                ind = dataset.traj_idx_num[sample]
                if os.path.isfile(os.path.join(dir_name, f"pred_{ind + 1:08d}.png")):
                    continue
                depth_img = features[1][-1,:,:,-1].numpy()


                pos_delta = label[0, ::10]/1.5
                rot_delta = cp.array([1, 0, 0, 0])
                state = cp.hstack((pos_delta, rot_delta))
                img_pred = predictor.pred_depth(depth_img, state)
                cv2.imwrite(os.path.join(dir_name, f"pred_{ind + 1:08d}.png"), img_pred)

                if sample == samples[0]:
                    print(f"Experiment_folder: {dir_name}")
                    mov_dir = 1
                    try:
                        depth_img = cv2.imread(f"{dir_name}/depth_{ind - 1:08d}.tif", cv2.IMREAD_ANYDEPTH)
                        dim = (nx, ny)
                        depth_img = cv2.resize(depth_img, dim)
                        depth_img = np.array(depth_img, dtype=np.float32)
                        depth_img = depth_img / (80)  # depth in (0.255)
                    except:
                        "First frame keep im0 for pred"
                        depth_img = depth_img
                        mov_dir = -1
                    try:
                        odometry = np.loadtxt(f'{dir_name}/../odometry.csv', skiprows=(ind), delimiter=',')
                        pos_delta0 = mov_dir * np.diff(odometry[:2, 1:4], axis=0).squeeze() + (np.random.random(3)-0.5)/10
                    except:
                        "First frame extrapolate pos_d"
                        pos_delta0 = mov_dir * label[0, ::10] / 15 + (np.random.random(3)-0.5)/10
                    state = cp.hstack((pos_delta0, rot_delta))
                    img = predictor.pred_depth(depth_img, state)
                    cv2.imwrite(f"{dir_name}/pred_{ind:08d}.png", img)
            ind_min = ind_max
    print("FINISHED")


if __name__ == "__main__":
    main()
