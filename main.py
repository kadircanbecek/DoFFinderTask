import argparse
from utils import *

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument("num_of_images", metavar="N", type=int, help="number of images to process")
args = parser.parse_args()
num_of_imgs = args.num_of_images
print(num_of_imgs)
imgs = [cv2.imread("img{0}.png".format(i + 1)) for i in range(num_of_imgs)]

image_size = imgs[0].shape[:2][::-1]

cam_matrix = find_intrinsic_matrix(image_size)
trajectories, rotation_translation = compute_relative_pose(num_of_imgs, imgs, cam_matrix)

plot_and_save_results(rotation_translation, trajectories)
