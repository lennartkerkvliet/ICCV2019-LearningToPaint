import torch
import numpy as np


def rgb_to_hex(value):
    rgb = value * 255
    return "#%02x%02x%02x" % tuple(rgb.astype(int))

def render_paths(paths, canvas, width=128):
    stroke = 1 - torch.from_numpy(np.array([path.draw() for path in paths]))
    stroke = stroke.view(-1, width, width, 1)

    colors = np.array([path.color[::-1] for path in paths])
    color_stroke = stroke * torch.from_numpy(colors).view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)

    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res
    
def render_svg(paths, width=128):
    svgstring = "<svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">".format(width * 2, width * 2)
    svgstring += "<rect width=\"{}\" height=\"{}\" fill=\"black\"/>".format(width * 2, width * 2)

    for path in paths:
        svgstring += path.shape_svg()  

    svgstring += "</svg>"
    return svgstring

def generate_stroke(curve, start_width, end_width, fill="black", opacity=1.0):
    plus = offset_curve(curve, start_width, end_width)
    min = offset_curve(curve, -start_width, -end_width)

    half1, half2 = offset_circle(plus[0], min[0], flip=True)
    half3, half4 = offset_circle(min[2], plus[2], flip=True)

    p0, p1, p2 = half1
    path = "M {} {} Q {} {}, {} {}".format(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
    for (p0, p1, p2) in [half2, min, half3, half4, np.flip(plus, axis=0)]:
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
    p0 = np.array(curve[0])
    p1 = np.array(curve[1])
    p2 = np.array(curve[2])

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


def offset_circle(p0, p1, flip=False):
    n0 = (p1 - p0) / 2
    n1 = orthagonal(n0, flip=flip)

    x1 = p0 + n1
    x2 = p0 + n0 + n1
    x3 = p1 + n1    

    return [(p0, x1, x2), (x2, x3, p1)]