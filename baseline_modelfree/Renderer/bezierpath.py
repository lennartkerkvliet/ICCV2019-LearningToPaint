import cv2
import numpy as np
from .render import rgb_to_hex
from .stroke import generate_stroke
from cairosvg import svg2png


class BezierPath:
    def __init__(self, f, width=128):
        x0, y0, x1, y1, x2, y2, z0, z2, w0, w2, b, g, r = f
        self.width = width

        def normal(x, width):
            return (int)(x * (width - 1) + 0.5)

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
        self.color = np.array([r, g, b]).astype('float32')

    def draw(self):
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

    def shape_svg(self):
        color = rgb_to_hex(self.color)
        return generate_stroke(self, self.z0, self.z2, color, max(self.w0, self.w2))

    # def split(self, t):
    #     q0 = self.lerp(start=p0, end=p1, t)
    #     q1 = self.lerp(start=p1, end=p2, t)
    #     q2 = self.lerp(start=p2, end=p3, t)
    #     r0 = self.lerp(start=q0, end=q1, t)
    #     r1 = self.lerp(start=q1, end=q2, t)
    #     s  = self.lerp(start=r0, end=r1, t)
    #     return (BezierPath(p0, q0, r0, s), BezierPath(s, r1, q2, p3))

    # def lerp(start, end, t):
    #     return start * (1.0 - t) + end * t

    def draw_svg(self):
        svgstring = "<svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">".format(self.width * 2, self.width * 2)
        svgstring += self.shape_svg()
        svgstring += "</svg>"

        image = svg2png(bytestring=svgstring, write_to=None)
        nparr = np.frombuffer(image, np.uint8)
        canvas = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        adjusted = canvas[:,:,3].astype('float32') / 255
        return 1 - cv2.resize(adjusted, dsize=(self.width, self.width))
