import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_intrinsic_matrix(image_size):
    # Load 2D 3D pairs for camera
    vr2d = np.load("vr2d.npy")
    image_points = vr2d.reshape(1, len(vr2d), -1)
    vr3d = np.load("vr3d.npy")
    object_points = vr3d.reshape(1, len(vr3d), -1)

    # Prepare initial guess for camera intrinsic matrix using prior info
    intrinsic_initial = np.eye(3)
    intrinsic_initial[0, 0] = 100
    intrinsic_initial[1, 1] = 100
    intrinsic_initial[0, 2] = 960
    intrinsic_initial[1, 2] = 540
    dist_coeff = np.zeros((14,), dtype=np.float32)

    # Calibrate using prior information and ground truth pairs
    _, cam_matrix, _, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        intrinsic_initial,
        dist_coeff,
        flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO +
               cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_S1_S2_S3_S4 + cv2.CALIB_FIX_K1 +
               cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3)
    )
    return cam_matrix


def find_rotation_translation(sift, cam_matrix, ref_points, second_img, MIN_MATCH_COUNT=0):

    kp1, des1 = ref_points
    # Detect KP and Descriptions from current image
    kp2, des2 = sift.detectAndCompute(second_img, None)
    # match the keypoints and descriptors from SIFT with Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches12 = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test to filter results
    good = []
    for m, n in matches12:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # Find essential matrix for pose estimation
        E, mask_e = cv2.findEssentialMat(dst_pts, src_pts, cam_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # Extract pose(rotation & translation) from Essential matrix
        _, R, t, _ = cv2.recoverPose(E, dst_pts, src_pts, cameraMatrix=cam_matrix, mask=mask_e)
        return R, t, (kp2, des2)
    else:
        raise ValueError("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))


def compute_relative_pose(num_of_imgs, imgs, cam_matrix):
    # Create feature extractor
    sift = cv2.SIFT_create()

    # Detect KP and Descriptions from first image
    ref_points = sift.detectAndCompute(imgs[0], None)

    # Assume initial point of camera as 0,0,0
    start_point = np.zeros([3, 1])
    # Initialize trajectory list with initial value
    trajectories = [start_point]
    # Initialize list to track rotation and translation w.r.t. img1.png
    rotation_translation = []
    for i in range(1, num_of_imgs):
        # Find rotation and translation using epipolar geometry, also replace old refpoints with current one
        R_relative, t_relative, ref_points = find_rotation_translation(sift, cam_matrix, ref_points, imgs[i])
        # Find new coordinates using old coordinates
        path_new = R_relative @ trajectories[-1] + t_relative
        if not rotation_translation:
            rotation_translation.append((R_relative, t_relative))
        else:
            # Compute rotation and translation with respect to img1
            R_old, t_old = rotation_translation[-1]
            # This formula is derived from R_relative@Point+t_relative = R_new@point_img1 + t_new
            # = R_relative@(R_old@point_img1 + t_old) + t_relative
            # = (R_relative@R_old)@point_img1 + (R_relative@t_old+t_relative)
            R_new = R_relative @ R_old
            t_new = R_relative @ t_old + t_relative
            rotation_translation.append((R_new, t_new))
        trajectories.append(path_new)
    return np.array(trajectories).reshape(num_of_imgs, -1), rotation_translation


def plot_and_save_results(rotation_translation, trajectories):
    # Save rotation and translation for every image to a txt file
    for i, (R, t) in enumerate(rotation_translation):
        with open("results/textresults/Rotation_translation_of_Image_{0}_from_Image_1.txt".format(i + 2), "w") as f:
            f.write("Rotation:\n")
            R_list = R.tolist()
            for row in R_list:
                f.write(" ".join(map(str, row)))
                f.write("\n")
            f.write("Translation:\n")
            f.write(" ".join(map(str, t.tolist())))
            f.write("\n")
    # Plot trajectory for x-axis only
    x_axis_for_plot = np.int32(range(len(trajectories))) + 1
    plt.plot(x_axis_for_plot, trajectories[:, 0], label='Camera Trajectory (X plane)')
    for i, traj in enumerate(trajectories):
        plt.scatter(x_axis_for_plot[i], traj[0], label="Image-{}".format(i + 1))
    plt.legend()
    plt.xlabel("Image Number")
    plt.ylabel("x")
    plt.xticks(x_axis_for_plot)
    plt.savefig("results/imgs/x-trajectory.png")
    plt.close("all")

    # Plot trajectory for y-axis only
    plt.plot(x_axis_for_plot, trajectories[:, 1], label='Camera Trajectory (Y plane)')
    for i, traj in enumerate(trajectories):
        plt.scatter(x_axis_for_plot[i], traj[1], label="Image-{}".format(i + 1))
    plt.legend()
    plt.xlabel("Image Number")
    plt.ylabel("y")
    plt.xticks(x_axis_for_plot)
    plt.savefig("results/imgs/y-trajectory.png")
    plt.close("all")

    # Plot trajectory for z-axis only
    plt.plot(x_axis_for_plot, trajectories[:, 2], label='Camera Trajectory (Z plane)')
    for i, traj in enumerate(trajectories):
        plt.scatter(x_axis_for_plot[i], traj[2], label="Image-{}".format(i + 1))
    plt.legend()
    plt.xlabel("Image Number")
    plt.ylabel("z")
    plt.xticks(x_axis_for_plot)
    plt.savefig("results/imgs/z-trajectory.png")
    plt.close("all")

    # Plot trajectory for all axes in 3D
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')
    for i, traj in enumerate(trajectories):
        ax.scatter3D(traj[0], traj[1], traj[2], label="Image-{}".format(i + 1))
    ax.plot(trajectories[:, 0], trajectories[:, 1].tolist(), trajectories[:, 2].tolist(),
            label='Camera Trajectory')
    ax.legend()
    ax.set_xlim([min(trajectories[:, 0]) - 0.1, max(trajectories[:, 0] + 0.1)])
    ax.set_ylim([min(trajectories[:, 1]) - 0.1, max(trajectories[:, 1] + 0.1)])
    ax.set_zlim([min(trajectories[:, 2]) - 0.1, max(trajectories[:, 2] + 0.1)])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig("results/imgs/3d-trajectory.png")
    plt.close("all")
