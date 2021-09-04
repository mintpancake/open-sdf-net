import numpy as np
import cv2
import time

from util import ensure_dir

CANVAS_SIZE = (800, 800)  # Do not change
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

DATA_PATH = '../curves/raw_data/'
IMAGE_PATH = '../curves/raw_images/'


# The class is adapted from
# https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python/37235130
class CurveDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name  # Name for our window

        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = []  # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print(f'Adding point #{len(self.points)} with position ({x},{y})')
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print(f"Completing curve with {len(self.points)} points")
            self.done = True

    def run(self, bg_path=''):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # TODO
        if bg_path != '':
            background = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
            background = cv2.resize(background, CANVAS_SIZE, interpolation=cv2.INTER_LINEAR)
        else:
            background = np.zeros(CANVAS_SIZE, np.uint8)

        while not self.done:
            # This is our drawing loop, we just continuously draw new shape_images
            # and show them in the named window
            canvas = np.zeros(CANVAS_SIZE, np.uint8)
            canvas += background
            if len(self.points) > 0:
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27:  # ESC hit
                self.done = True

        # User finished entering the polygon points, so let's make the final drawing
        canvas = np.zeros(CANVAS_SIZE, np.uint8)
        if len(self.points) > 0:
            cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 2)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return self.points, canvas


class DataSaver(object):
    def __init__(self, points, image, data_path=DATA_PATH, image_path=IMAGE_PATH, save_name=''):
        self.points = points
        self.image = image
        self.data_path = data_path
        self.image_path = image_path
        if save_name == '':
            self.save_name = f'curve_{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'
        else:
            self.save_name = save_name

    def save(self):
        ensure_dir(self.data_path)
        f = open(f'{self.data_path}{self.save_name}.txt', 'w')
        for point in self.points:
            # Normalize (x,y) to (0,1)
            x = np.double(np.double(point[0]) / np.double(CANVAS_SIZE[0]))
            y = np.double(np.double(point[1]) / np.double(CANVAS_SIZE[1]))
            f.write(f'{x} {y}\n')
        f.close()
        print(f'Data path = {self.data_path}{self.save_name}.txt')

        ensure_dir(self.image_path)
        cv2.imwrite(f'{self.image_path}{self.save_name}.png', self.image)
        print(f'Image path = {self.image_path}{self.save_name}.png')


if __name__ == '__main__':
    print('Enter save name (skip with enter key): ')
    name = input()
    print('Enter background image path (skip with enter key): ')
    path = input()

    drawer = CurveDrawer('Left click: Add vertices    Right click: Finish    Any key: Save data')
    points, image = drawer.run(bg_path=path)
    print(f'Curve = {points}')

    saver = DataSaver(points, image, save_name=name)
    saver.save()
    print('Done!')
