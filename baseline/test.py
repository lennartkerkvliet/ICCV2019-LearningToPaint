import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F

from DRL.actor import *
from Renderer.bezierpath import BezierPath
from Renderer.render import render_svg
from Renderer.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./model/Paint-run1/actor.pkl', type=str, help='Actor model')
parser.add_argument('--corner_radius', default=False, action='store_true', help='Supports corner radius')
parser.add_argument('--img', default='image/test.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
args = parser.parse_args()

T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)
img = cv2.imread(args.img, cv2.IMREAD_COLOR)
origin_shape = (img.shape[1], img.shape[0])

coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device) # Coordconv

def decode(x, canvas, width=128): # b * (10 + 3)
    x = x.view(-1, 10 + 3)
    paths = [BezierPath(f[:10], color=f[-3:], width=width) for f in x]
    stroke = 1 - torch.from_numpy(np.array([path.draw_svg(corner_radius=args.corner_radius) for path in paths]))
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res, paths

def small2large(x):
    # (d * d, width, width) -> (d * width, d * width)    
    x = x.reshape(1, 1, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(width, width, -1)
    return x

def large2small(x):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(1, width, 1, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(1, width, width, 3)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == width - 1 or ty == width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for k in range(width):
        img = smooth_pix(img, k, width - 1)
    for k in range(width):
        img = smooth_pix(img, width - 1, k)
    return img

def save_img(res, imgid):
    output = res.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    output = output[0]
    output = (output * 255).astype('uint8')
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)

def save_svg(res, imgid):
    file = open('output/vector_generated' + str(imgid) + '.svg', "w")
    file.write(res)
    file.close()

actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()

canvas = torch.ones([1, 3, width, width]).to(device)

patch_img = cv2.resize(img, (width, width))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img).to(device).float() / 255.

os.system('mkdir output')

with torch.no_grad():
    args.max_step = args.max_step // 2

    paths_res = []
    for i in range(args.max_step):
        stepnum = T * i / args.max_step
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        canvas, res, paths = decode(actions, canvas)
        print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))

        for j in range(5):
            paths_res.append(paths[j])
            svgstring = render_svg(paths_res, width=width, corner_radius=args.corner_radius)

            # save_img(res[j], args.imgid)
            save_svg(svgstring, args.imgid)
            args.imgid += 1
