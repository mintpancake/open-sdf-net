import numpy as np
import cv2
from renderer import plot_sdf
from curve import Curve
from util import ensure_dir, read_config
from transformer import transform

CURVE_PATH = '../curves/raw_data/'
CURVE_IMAGE_PATH = '../curves/raw_images/'

NORM_PATH = '../curves/normalized_data/'
NORM_IMAGE_PATH = '../curves/normalized_images/'

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'

SAMPLED_IMAGE_PATH = '../datasets/sampled_images/'
HEATMAP_PATH = '../results/true_heatmaps/'

CANVAS_SIZE = np.array([800, 800])  # Do not change
CURVE_COLOR = (255, 255, 255)
POINT_COLOR = (127, 127, 127)

CFG = read_config()


class ShapeSampler(object):
    def __init__(self,
                 curve_name,
                 curve_path=CURVE_PATH,
                 curve_image_path=CURVE_IMAGE_PATH,
                 norm_path=NORM_PATH,
                 norm_image_path=NORM_IMAGE_PATH,
                 train_data_path=TRAIN_DATA_PATH,
                 val_data_path=VAL_DATA_PATH,
                 sampled_image_path=SAMPLED_IMAGE_PATH,
                 split_ratio=CFG["split_ratio"],
                 show_image=False
                 ):
        """
        :param split_ratio: train / (train + val)
        :param show_image: Launch a windows showing sampled image
        """

        self.curve_name = curve_name
        self.curve_path = curve_path
        self.curve_image_path = curve_image_path

        self.norm_path = norm_path
        self.norm_image_path = norm_image_path

        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.sampled_image_path = sampled_image_path

        self.split_ratio = split_ratio
        self.show_image = show_image

        self.curve = Curve()
        self.sampled_data = None
        self.train_data = None
        self.val_data = None

    def run(self):
        self.load()
        self.normalize()
        self.sample()
        self.save()

    def load(self):
        self.curve.load(self.curve_path, self.curve_name)

    def normalize(self, scope=0.4):
        """
        :param scope: Should be less than 0.5
        """
        if self.curve.v is None:
            return
        # Calculate the center of gravity
        # and translate the shape
        g = np.mean(self.curve.v, axis=0)
        trans_curve = self.curve.v - g
        # Calculate the farthest away point
        # and scale the shape so that it is bounded by the unit circle
        max_dist = np.max(np.linalg.norm(trans_curve, axis=1))
        scaling_factor = scope / max_dist
        trans_curve *= scaling_factor
        self.curve.set_v(trans_curve)

        # Save normalized data
        ensure_dir(self.norm_path)
        f = open(f'{self.norm_path}{self.curve_name}.txt', 'w')
        for datum in self.curve.v:
            f.write(f'{datum[0]} {datum[1]}\n')
        f.close()
        print(f'Normalized data path = {self.norm_path}{self.curve_name}.txt')

        scaled_v = np.around(self.curve.v * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)
        norm = np.zeros(CANVAS_SIZE, np.uint8)
        cv2.polylines(norm, scaled_v[np.newaxis, :, :], False, CURVE_COLOR, 2)
        ensure_dir(self.norm_image_path)
        cv2.imwrite(f'{self.norm_image_path}{self.curve_name}.png', norm)
        print(f'Normalized image path = {self.norm_image_path}{self.curve_name}.png')

        # Plot_sdf
        transform(self.curve_name)
        plot_sdf(self.curve.sdf, self.curve, self.curve_name, HEATMAP_PATH, self.norm_image_path,
                 device='cpu', is_net=False, show=self.show_image)

    def sample(self, m=CFG["m"], n=CFG["n"], var=CFG["var"]):  # 5000 2000
        """
        :param m: number of points sampled on the boundary
                  each boundary point generates 2 samples
        :param n: number of points sampled uniformly in the unit circle
        :param var: two Gaussian variances used to transform boundary points
        """

        if self.curve.v is None:
            return

        # Do uniform sampling
        # Use polar coordinate
        r = np.sqrt(np.random.uniform(0, 1, size=(n, 1))) / 2
        t = np.random.uniform(0, 2 * np.pi, size=(n, 1))
        # Transform to Cartesian coordinate
        uniform_points = np.concatenate((r * np.cos(t), r * np.sin(t)), axis=1)

        # Do Gaussian sampling
        # Distribute points to each edge weighted by length
        total_length = 0
        edge_length = np.zeros(len(self.curve.e), dtype=np.float64)
        for i in range(len(self.curve.e)):
            length = np.linalg.norm(self.curve.e[i, 1] - self.curve.e[i, 0])
            edge_length[i] = length
            total_length += length
        edge_portion = edge_length / total_length
        edge_portion *= m
        edge_num = np.around(edge_portion).astype(int)

        # Do sampling on edges
        direction = (self.curve.e[0, 1] - self.curve.e[0, 0])
        d = np.random.uniform(0, 1, size=(edge_num[0], 1))
        boundary_points = self.curve.e[0, 0] + d * direction
        for i in range(1, len(self.curve.e)):
            direction = (self.curve.e[i, 1] - self.curve.e[i, 0])
            d = np.random.uniform(0, 1, size=(edge_num[i], 1))
            boundary_points = np.concatenate((boundary_points, self.curve.e[i, 0] + d * direction), axis=0)

        # Perturbing boundary points
        noise_1 = np.random.normal(loc=0, scale=np.sqrt(var[0]), size=boundary_points.shape)
        noise_2 = np.random.normal(loc=0, scale=np.sqrt(var[1]), size=boundary_points.shape)
        gaussian_points = np.concatenate((boundary_points + noise_1, boundary_points + noise_2), axis=0)

        # Merge uniform and Gaussian points
        sampled_points = np.concatenate((uniform_points, gaussian_points), axis=0)
        self.sampled_data = self.calculate_sdf(sampled_points)

        # Split sampled data into train dataset and val dataset
        train_size = int(len(self.sampled_data) * self.split_ratio)
        choice = np.random.choice(range(self.sampled_data.shape[0]), size=(train_size,), replace=False)
        ind = np.zeros(self.sampled_data.shape[0], dtype=bool)
        ind[choice] = True
        rest = ~ind
        self.train_data = self.sampled_data[ind]
        self.val_data = self.sampled_data[rest]
        # self.train_data = self.sampled_data[:train_size]
        # self.val_data = self.sampled_data[train_size:]

    def calculate_sdf(self, points):
        if self.curve.v is None:
            return

        # Add a third column for storing sdf
        data = np.concatenate((points, np.zeros((points.shape[0], 1))), axis=1)
        data[:, 2] = np.apply_along_axis(self.curve.sdf, 1, data[:, :2])
        return data

    def save(self):
        if self.curve.v is None:
            return

        # Split sampled data into train dataset and val dataset
        train_size = int(len(self.sampled_data) * self.split_ratio)
        choice = np.random.choice(range(self.sampled_data.shape[0]), size=(train_size,), replace=False)
        ind = np.zeros(self.sampled_data.shape[0], dtype=bool)
        ind[choice] = True
        rest = ~ind
        self.train_data = self.sampled_data[ind]
        self.val_data = self.sampled_data[rest]

        # Save data to .txt
        ensure_dir(self.train_data_path)
        f = open(f'{self.train_data_path}{self.curve_name}.txt', 'w')
        for datum in self.train_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]}\n')
        f.close()
        ensure_dir(self.val_data_path)
        f = open(f'{self.val_data_path}{self.curve_name}.txt', 'w')
        for datum in self.val_data:
            f.write(f'{datum[0]} {datum[1]} {datum[2]}\n')
        f.close()
        print(f'Sampled data path = {self.train_data_path}{self.curve_name}.txt\n'
              f'                    {self.val_data_path}{self.curve_name}.txt')

        # Generate a sampled image
        canvas = np.zeros(CANVAS_SIZE, np.uint8)
        # Draw curve
        scaled_v = np.around(self.curve.v * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)
        cv2.polylines(canvas, scaled_v[np.newaxis, :, :], False, CURVE_COLOR, 2)
        # Draw points
        for i, datum in enumerate(self.sampled_data):
            point = np.around(datum[:2] * CANVAS_SIZE + CANVAS_SIZE / 2).astype(int)
            cv2.circle(canvas, point, 1, POINT_COLOR, -1)
            # if i % 100 == 0:
            #     radius = np.abs(np.around(datum[2] * CANVAS_SIZE[0]).astype(int))
            #     cv2.circle(canvas, point, radius, POINT_COLOR)

        # Store and show
        ensure_dir(self.sampled_image_path)
        cv2.imwrite(f'{self.sampled_image_path}{self.curve_name}.png', canvas)
        print(f'Sampled image path = {self.sampled_image_path}{self.curve_name}.png')

        if not self.show_image:
            return

        cv2.imshow('Sampled Image', canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Enter curve name:')
    name = input()
    sampler = ShapeSampler(name)
    print('Sampling...')
    sampler.run()
    print('Done!')
