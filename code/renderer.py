import os
import numpy as np
import cv2
import torch
from net import SDFNet
from curve import Curve
from gradient import Gradient
from util import ensure_dir

MODEL_PATH = '../models/'
DATA_PATH = '../curves/normalized_data/'
MASK_PATH = '../curves/normalized_images/'
TRAINED_PATH = '../results/trained_heatmaps/'
TRUE_PATH = '../results/true_heatmaps/'
GRID_PATH = '../curves/sdf_grid/'
MEDIAL_PATH = '../curves/raw_medial_images/'
ERROR_PATH = '../results/errors/'


# Adapted from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
def plot_sdf(sdf_func, curve, name, res_path, mask_path=MASK_PATH, levels=(0,),
             device='cpu', is_net=False, show=False):
    # Sample the 2D domain as a regular grid
    # img_size = 800
    low = -0.5
    high = 0.5
    grid_size = 800
    margin = 8e-3  # 8e-3
    # gradient_margin = np.inf
    hazard_kernel = 7
    delta = curve.delta  # Normalizing distance

    # Load ground truth SDF
    gt_sdf_func = curve.sdf

    grid = np.linspace(low, high, grid_size + 1)[:-1]

    grid_path = f'{GRID_PATH}{name}.npy'
    if os.path.exists(grid_path):
        gt_sdf_map = np.load(grid_path)
    else:
        gt_sdf_map = [[gt_sdf_func(np.float_([x_, y_]))
                       for x_ in grid] for y_ in grid]
        gt_sdf_map = np.array(gt_sdf_map, dtype=np.float64)
        ensure_dir(GRID_PATH)
        np.save(grid_path, gt_sdf_map)
        print(f'SDF grid path = {grid_path}')

    if not is_net:
        sdf_map = gt_sdf_map
    else:
        grid_points = [[[x, y] for x in grid] for y in grid]
        sdf_map = []
        sdf_func.eval()
        with torch.no_grad():
            for row in grid_points:
                row = torch.Tensor(row).to(device)
                row_sdf = sdf_func(row).detach().cpu().numpy()
                sdf_map.append(row_sdf)
        sdf_map = np.array(sdf_map)
        sdf_map = np.reshape(sdf_map, [grid_size, grid_size])

    max_norm = np.max(np.abs(sdf_map)) if delta == 0 else delta
    heat_map = sdf_map / max_norm * 127.5 + 127.5
    heat_map = np.minimum(heat_map, 255)
    heat_map = np.maximum(heat_map, 0)

    # Generate a heat map
    heat_map = np.uint8(heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)

    # Plot predicted level sets
    level_sets = np.zeros(sdf_map.shape, dtype=bool)
    for level in levels:
        low_set = sdf_map > level - margin
        high_set = sdf_map < level + margin
        level_set = low_set & high_set
        level_sets = level_sets | level_set

    # Filter large gradient point
    # if is_net and gradient_margin != np.inf:
    #     zero_idx = np.array(np.where(zero_pos))
    #     for i in range(zero_idx.shape[1]):
    #         idx_1, idx_2 = zero_idx[:, i]
    #         x, y = grid[idx_1], grid[idx_2]
    #         xy = torch.Tensor([[x, y]])
    #         if Gradient.evaluate(sdf_func.to('cpu'), xy) > 1 + gradient_margin:
    #             zero_pos[idx_1, idx_2] = False

    pred = np.where(level_sets[:, :, np.newaxis], np.zeros([grid_size, grid_size, 3], np.uint8),
                    np.ones([grid_size, grid_size, 3], np.uint8) * 255)
    heat_map = np.minimum(heat_map, pred)

    # Plot mask
    # Truncate
    mask = cv2.inRange(gt_sdf_map, -delta, delta)
    # Eliminate end points
    grid_points = [[x, y] for x in grid for y in grid]
    candidates = [p for p in grid_points
                  if np.linalg.norm(np.array(p) - curve.v[0]) <= delta
                  or np.linalg.norm(np.array(p) - curve.v[-1]) <= delta]
    for p in candidates:
        near = curve.nearest(np.array(p))
        if (near == curve.v[0]).all() or (near == curve.v[-1]).all():
            idx1, idx2 = np.where(grid == p[0]), np.where(grid == p[1])
            # Endpoints
            mask[idx2, idx1] = 0

    # Eliminate hazard points
    medial_img = cv2.imread(f'{MEDIAL_PATH}{name}.png', cv2.IMREAD_GRAYSCALE)
    medial = np.ones([grid_size, grid_size])
    medial[medial_img == 0] = 0

    y_indices, x_indices = np.where(medial == 1)
    for i in range(len(y_indices)):
        y_idx, x_idx = y_indices[i], x_indices[i]
        kernel_sdf = []
        h = int((hazard_kernel - 1) / 2)
        y_low, y_high = max(0, y_idx - h), min(grid_size - 1, y_idx + h + 1)
        x_low, x_high = max(0, x_idx - h), min(grid_size - 1, x_idx + h + 1)
        for y_ in range(y_low, y_high):
            for x_ in range(x_low, x_high):
                kernel_sdf.append(sdf_map[y_, x_])
        kernel_sdf = np.array(kernel_sdf)
        kernel_sdf_sign = np.sign(kernel_sdf)
        if (not np.all(kernel_sdf_sign == kernel_sdf_sign[0])) \
                and (np.max(kernel_sdf) - np.min(kernel_sdf) >= hazard_kernel / grid_size * 1.415):
            for y_ in range(y_low, y_high):
                for x_ in range(x_low, x_high):
                    # Close points
                    pass
                    # mask[y_, x_] = 0

    heat_map[mask == 0] = (255, 255, 255)

    # Calculate L1 norm
    if is_net:
        num = np.count_nonzero(mask)
        gt_sdf_map[mask == 0] = 0
        sdf_map[mask == 0] = 0
        err = np.abs(gt_sdf_map - sdf_map)
        avg_err = np.sum(err) / num
        max_err = np.max(err)

        ensure_dir(ERROR_PATH)
        f = open(f'{ERROR_PATH}{name}.txt', 'w')
        f.write(f'Avg error: {avg_err}\n')
        f.write(f'Max error: {max_err}\n')
        f.close()
        print(f'Error data path = {ERROR_PATH}{name}.txt')

    # Scale to canvas size
    # scale = int(img_size / grid_size)
    # heat_map = np.kron(heat_map, np.ones((scale, scale, 1))).astype(np.uint8)
    # heat_map = cv2.resize(heat_map, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Plot true boundary
    true = cv2.imread(f'{MASK_PATH}{name}.png')
    heat_map = np.maximum(heat_map, true)

    ensure_dir(res_path)
    cv2.imwrite(f'{res_path}{name}.png', heat_map)
    print(f'Heatmap path = {res_path}{name}.png')

    if not show:
        return

    cv2.imshow('SDF Map', heat_map)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mode = ''
    while mode != 'trained' and mode != 'true':
        print('Choose mode (trained/true):')
        mode = input()

    print('Enter curve name:')
    name = input()

    print('Enter level sets: (use space to separate)')
    input_levels = input()
    levels = [float(n) for n in input_levels.split()]

    curve = Curve()
    curve.load(DATA_PATH, name)

    if mode == 'trained':
        net = True
        path = TRAINED_PATH
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device}!')
        sdf_func = SDFNet().to(device)
        if os.path.exists(f'{MODEL_PATH}{name}.pth'):
            sdf_func.load_state_dict(torch.load(f'{MODEL_PATH}{name}.pth'))
        else:
            print('Error: No trained data!')
            exit(-1)
    else:
        net = False
        path = TRUE_PATH
        device = 'cpu'
        sdf_func = curve.sdf

    print('Plotting results...')
    plot_sdf(sdf_func, curve, name, path, levels=levels, device=device, is_net=net, show=False)
    print('Done!')
