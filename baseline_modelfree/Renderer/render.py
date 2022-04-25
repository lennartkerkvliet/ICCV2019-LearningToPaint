import torch
import numpy as np


def render_paths(paths, canvas, width=128):
    stroke = 1 - torch.from_numpy(np.array([path.draw() for path in paths]))
    stroke = stroke.view(-1, width, width, 1)

    colors = np.array([path.color for path in paths])
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
    