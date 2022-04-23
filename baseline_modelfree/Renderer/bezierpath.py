import cv2
import numpy as np


def normal(x, width):
    return (int)(x * (width - 1) + 0.5)

class BezierPath:
    def __init__(self, f, width=128):
        x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
        self.width = width

        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        self.x0 = normal(x0, width * 2)
        self.x1 = normal(x1, width * 2)
        self.x2 = normal(x2, width * 2)
        self.y0 = normal(y0, width * 2)
        self.y1 = normal(y1, width * 2)
        self.y2 = normal(y2, width * 2)
        self.z0 = (int)(1 + z0 * width // 2)
        self.z2 = (int)(1 + z2 * width // 2)
        self.w0 = (float)(w0)
        self.w2 = (float)(w2)

    def draw(self):
        canvas = np.zeros([self.width * 2, self.width * 2]).astype('float32')
        tmp = 1. / 100
        for i in range(100):
            t = i * tmp
            x = (int)((1-t) * (1-t) * self.x0 + 2 * t * (1-t) * self.x1 + t * t * self.x2)
            y = (int)((1-t) * (1-t) * self.y0 + 2 * t * (1-t) * self.y1 + t * t * self.y2)
            z = (int)((1-t) * self.z0 + t * self.z2)
            w = (1-t) * self.w0 + t * self.w2
            cv2.circle(canvas, (y, x), z, w, -1)
        return 1 - cv2.resize(canvas, dsize=(self.width, self.width))