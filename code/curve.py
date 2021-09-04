import os
import numpy as np


# The classes are adapted from
# https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb


class Geometry(object):
    EPS = 1e-12

    @staticmethod
    def p2e_distance(e, p):
        a, b = e[0], e[1]
        res = np.minimum(np.linalg.norm(a - p), np.linalg.norm(b - p))
        if (np.linalg.norm(a - b) > Geometry.EPS
                and np.dot(p - a, b - a) > Geometry.EPS
                and np.dot(p - b, a - b) > Geometry.EPS):
            res = abs(np.cross(p - a, b - a) / np.linalg.norm(b - a))
        return res

    @staticmethod
    def intersect(e1, e2):
        dx0 = e1[1][0] - e1[0][0]
        dx1 = e2[1][0] - e2[0][0]
        dy0 = e1[1][1] - e1[0][1]
        dy1 = e2[1][1] - e2[0][1]
        p0 = dy1 * (e2[1][0] - e1[0][0]) - dx1 * (e2[1][1] - e1[0][1])
        p1 = dy1 * (e2[1][0] - e1[1][0]) - dx1 * (e2[1][1] - e1[1][1])
        p2 = dy0 * (e1[1][0] - e2[0][0]) - dx0 * (e1[1][1] - e2[0][1])
        p3 = dy0 * (e1[1][0] - e2[1][0]) - dx0 * (e1[1][1] - e2[1][1])
        return (p0 * p1 <= 0) & (p2 * p3 <= 0)  # Change to < if exclude overlapping case

    @staticmethod
    def closer_vertex(e, p):
        a, b = e[0], e[1]
        ap, bp = np.linalg.norm(a - p), np.linalg.norm(b - p)
        v = a if ap < bp else b
        if (np.linalg.norm(a - b) > Geometry.EPS
                and np.dot(p - a, b - a) > Geometry.EPS
                and np.dot(p - b, a - b) > Geometry.EPS):
            v = None
        return v

    @staticmethod
    def side(e, p):
        # Left: -    Right: +
        a, b = e[0], e[1]
        return np.sign(np.cross(b - a, p - a))


class Curve(object):
    MAX_DELTA = 0.1

    def __init__(self):
        self.v = None
        self.e = None
        self.delta = Curve.MAX_DELTA

    def set_v(self, v):
        self.v = np.array(v, dtype=np.float64)
        e = []
        for i in range(len(v) - 1):
            e.append([v[i], v[i + 1]])
        self.e = np.array(e, dtype=np.float64)
        # self.set_delta()

    def sdf(self, p):
        min_dist = np.inf
        min_idx = 0
        tip = False
        for i in range(len(self.e)):
            curr_dist = Geometry.p2e_distance(self.e[i], p)
            if curr_dist <= min_dist:
                if curr_dist == min_dist and i == min_idx + 1:
                    tip = True
                else:
                    tip = False
                    min_dist = curr_dist
                min_idx = i
        if tip:
            a, b, c = self.e[min_idx - 1, 0], self.e[min_idx - 1, 1], self.e[min_idx, 1]
            u, v = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
            d = b + u + v
            res = Geometry.side(np.array([b, d]), p) * min_dist
        else:
            res = Geometry.side(self.e[min_idx], p) * min_dist
        return res

    def nearest(self, p):
        min_dist = np.inf
        nearest_v = None
        for i in range(len(self.e)):
            dist = Geometry.p2e_distance(self.e[i], p)
            v = Geometry.closer_vertex(self.e[i], p)
            if dist <= min_dist:
                min_dist = dist
                nearest_v = v
        return nearest_v

    def set_delta(self):
        low = -0.5
        high = 0.5
        grid_size = 100
        grid = np.linspace(low, high, grid_size + 1)  # -0.5 to 0.5 (inclusive)
        sdf = [[self.sdf(np.float_([x, y])) for y in grid] for x in grid]
        dist = np.inf
        for i in range(len(grid) - 1):
            for j in range(len(grid) - 1):
                point_square = np.array([[grid[i], grid[j]], [grid[i + 1], grid[j]],
                                         [grid[i], grid[j + 1]], [grid[i + 1], grid[j + 1]]])
                sdf_square = np.array([sdf[i][j], sdf[i + 1][j], sdf[i][j + 1], sdf[i + 1][j + 1]])
                sign_square = np.sign(sdf_square)
                # Signs are different and they are not around true curve boundary
                if (not np.all(sign_square == sign_square[0])) \
                        and (np.max(sdf_square) - np.min(sdf_square) >= ((high - low) / grid_size) * 1.415):
                    # 1.415 is approximately sqrt(2)
                    # Rule out points near end points
                    end_point = False
                    for p in point_square:
                        near = self.nearest(p)
                        if (near == self.v[0]).all() or (near == self.v[-1]).all():
                            end_point = True
                            break
                    if not end_point:
                        dist = np.minimum(np.max(np.abs(sdf_square)), dist)
        self.delta = np.minimum(dist, Curve.MAX_DELTA)
        print(f'Truncation distance: {self.delta}')

    def load(self, path, name):
        if not os.path.exists(f'{path}{name}.txt'):
            print('Error: No curve data!')
            exit(-1)

        vertices = []
        f = open(f'{path}{name}.txt', 'r')
        line = f.readline()
        while line:
            x, y = map(lambda n: np.float64(n), line.strip('\n').split(' '))
            vertices.append([x, y])
            line = f.readline()
        f.close()
        self.set_v(np.array(vertices, dtype=np.float64))
