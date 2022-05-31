import numpy as np


def rgb_to_hex(value):
    rgb = value * 255
    return "#%02x%02x%02x" % tuple(rgb.astype(int))

def generate_stroke(curve, start_width, end_width, fill="black", opacity=1.0, corner_radius=1.0):
    if curve.y0 == curve.y1 == curve.y2 and curve.x0 == curve.x1 == curve.x2:
        return "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"/>".format(curve.y0, curve.x0, max(start_width, end_width))

    plus = offset_curve(curve, start_width, end_width)
    min = offset_curve(curve, -start_width, -end_width)

    half1 = offset_circle(plus[0], min[0], corner_radius, flip=True)
    half2 = offset_circle(min[2], plus[2], corner_radius, flip=True)

    p0, p1, p2 = min
    path = "M {} {} Q {} {}, {} {}".format(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
    for (p0, p1, p2) in half2 + [np.flip(plus, axis=0)] + half1:
        path += "Q {} {}, {} {}".format(p1[0], p1[1], p2[0], p2[1])
    path += "Z"
    return "<path d=\"{}\" fill=\"{}\" fill-opacity=\"{}\"/>".format(path, fill, opacity)

def orthagonal(vec, flip=False):
    if flip:
        return -np.array([-vec[1], vec[0]])
    else:
        return np.array([-vec[1], vec[0]])

def resize(curve, length=1):
    size = np.linalg.norm(curve)
    if size == 0:
        return curve
    return curve / np.linalg.norm(curve) * length

def offset_curve(curve, offset1, offset2):
    p0 = np.array([curve.y0, curve.x0])
    p1 = np.array([curve.y1, curve.x1])
    p2 = np.array([curve.y2, curve.x2])

    d0 = np.linalg.norm(p1 - p0)
    d1 = np.linalg.norm(p1 - p2)

    t0 = d0 / (d0 + d1) * 2
    t1 = d1 / (d0 + d1) * 2

    # calculate orthagonal vector
    n0 = resize(orthagonal(p1 - p0))
    n1 = resize(orthagonal(p1 - p2, flip=True))
    n = t0 * n0 + t1 * n1

    x0 = p0 + n0 * offset1
    x1 = p1 + (t0 * n0 * offset1 * 2 + t1 * n1 * offset2 * 2) / np.dot(n, n)
    x2 = p2 + n1 * offset2

    return (x0, x1, x2)

def offset_circle(p0, p1, corner_radius=1.0, flip=False):
    n0 = (p1 - p0) / 2 * corner_radius
    n1 = orthagonal(n0, flip=flip)

    if corner_radius == 0.0:
        return [(p0, p0 + n0, p1)]

    x1 = p0 + n1
    x2 = p0 + n0 + n1
    x3 = p1 - n0 + n1 
    x4 = p1 + n1
    x5 = x2 + (x3 - x2) / 2

    return [(p0, x1, x2), (x2, x5, x3), (x3, x4, p1)]