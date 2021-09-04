import os

import cv2
from skimage import morphology
import numpy as np
from util import ensure_dir

SRC_IMG_PATH = '../curves/normalized_images/'
# SKELETON_PATH = '../curves/skeleton_data/'
# SKELETON_IMG_PATH = '../curves/skeleton_images/'
MEDIAL_PATH = '../curves/medial_data/'
MEDIAL_IMG_PATH = '../curves/medial_images/'
DISTANCE_PATH = '../curves/distance_data/'
DISTANCE_IMG_PATH = '../curves/distance_images/'
RAW_MEDIAL_IMG_PATH = '../curves/raw_medial_images/'


# def skeleton_transform(name, img, raw, show):
#     _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
#     binary[binary == 255] = 1
#     skeleton0 = morphology.skeletonize(binary)
#     skeleton = skeleton0.astype(np.uint8) * 255
#     skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
#     res = np.minimum(skeleton + raw, 255)
#     ensure_dir(SKELETON_IMG_PATH)
#     cv2.imwrite(f'{SKELETON_IMG_PATH}{name}.png', res)
#     print(f'Skeleton image path = {SKELETON_IMG_PATH}{name}.png')
#     if show:
#         cv2.imshow('Skeleton Transform', res)
#         cv2.waitKey()
#         cv2.destroyAllWindows()

def medial_axis_transform(name, img, raw, show):
    _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255
    dist_on_skel = cv2.applyColorMap(dist_on_skel, cv2.COLORMAP_BONE)
    ensure_dir(RAW_MEDIAL_IMG_PATH)
    cv2.imwrite(f'{RAW_MEDIAL_IMG_PATH}{name}.png', dist_on_skel)
    print(f'Raw medial axis image path = {RAW_MEDIAL_IMG_PATH}{name}.png')
    res = np.minimum(dist_on_skel + raw, 255)
    ensure_dir(MEDIAL_IMG_PATH)
    cv2.imwrite(f'{MEDIAL_IMG_PATH}{name}.png', res)
    print(f'Medial axis image path = {MEDIAL_IMG_PATH}{name}.png')
    if show:
        cv2.imshow('Medial Axis Transform', res)
        cv2.waitKey()
        cv2.destroyAllWindows()


def distance_transform(name, img, raw, show):
    _, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    distance = cv2.normalize(distance, distance, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    distance = cv2.applyColorMap(distance, cv2.COLORMAP_BONE)
    res = np.minimum(distance + raw, 255)
    ensure_dir(DISTANCE_IMG_PATH)
    cv2.imwrite(f'{DISTANCE_IMG_PATH}{name}.png', res)
    print(f'Distance image path = {DISTANCE_IMG_PATH}{name}.png')
    if show:
        cv2.imshow('Distance Transform', res)
        cv2.waitKey()
        cv2.destroyAllWindows()


def transform(name):
    if not os.path.exists(f'{SRC_IMG_PATH}{name}.png'):
        print('Error: No normalized image!')
        exit(-1)
    img = cv2.imread(f'{SRC_IMG_PATH}{name}.png', cv2.IMREAD_GRAYSCALE)
    w, h = img.shape[:2]

    corners = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    cv2.polylines(img, np.array([[corners]]), True, (255, 255, 255), 1)

    raw = np.zeros([w, h, 3], dtype=np.uint8)
    loc = img != 0
    raw[loc] = np.array([0, 0, 255])

    medial_axis_transform(name, img, raw, False)
    distance_transform(name, img, raw, False)


if __name__ == '__main__':
    print('Enter curve name:')
    name = input()
    transform(name)
    print('Done!')
