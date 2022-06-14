import cv2
import numpy as np
from .stroke_gen import generate_stroke
from cairosvg import svg2png


class BezierPath:
    def __init__(self, f, color=[0, 0, 0], width=128):
        x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
        # x0, y0, x1, y1, x2, y2, z0, z2, w0, w2, b, g, r = f
        self.width = width

        def normal(x, width):
            return (int)(x * (width - 1) + 0.5)

        x1 = x0 + (x2 - x0) * x1
        y1 = y0 + (y2 - y0) * y1
        self.x0 = normal(x0, width)
        self.x1 = normal(x1, width)
        self.x2 = normal(x2, width)
        self.y0 = normal(y0, width)
        self.y1 = normal(y1, width)
        self.y2 = normal(y2, width)
        self.z0 = (int)(1 + z0 * (width // 4))
        self.z2 = (int)(1 + z2 * (width // 4))
        self.w0 = (float)(w0)
        self.w2 = (float)(w2)

        self.w0 = (float)(w0)
        self.w2 = (float)(w2)

        self.color = np.flip(np.array(color)).astype('float32')

    def draw_opencv(self):
        canvas = np.zeros([self.width * 2, self.width * 2]).astype('float32')
        tmp = 1. / 100
        for i in range(100):
            t = i * tmp
            x = (int)((1-t) * (1-t) * self.x0 + 2 * t * (1-t) * self.x1 + t * t * self.x2)
            y = (int)((1-t) * (1-t) * self.y0 + 2 * t * (1-t) * self.y1 + t * t * self.y2)
            radius = (int)((1-t) * self.z0 + t * self.z2)
            color = (1-t) * self.w0 + t * self.w2

            cv2.circle(canvas, (y, x), radius, color, -1)
        return 1 - cv2.resize(canvas, dsize=(self.width, self.width))

    def draw_svg(self, corner_radius=True):
        # color = rgb_to_hex(self.color)

        svgstring = "<svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">".format(self.width, self.width)
        if corner_radius:
            svgstring += generate_stroke(self, self.z0, self.z2, corner_radius=self.w2)
        else:
            svgstring += generate_stroke(self, self.z0, self.z2)
        svgstring += "</svg>"

        image = svg2png(bytestring=svgstring, write_to=None)
        nparr = np.frombuffer(image, np.uint8)
        canvas = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        adjusted = canvas[:,:,3].astype('float32') / 255
        return 1 - adjusted