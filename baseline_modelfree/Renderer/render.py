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

