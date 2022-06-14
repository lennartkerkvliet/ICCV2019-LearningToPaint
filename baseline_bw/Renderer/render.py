from .bezierpath import BezierPath
from .stroke_gen import generate_stroke


def render_svg(paths, width=128):
    svgstring = "<svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">".format(width, width)
    for path in paths:
        svgstring += shape_svg(path)  
    svgstring += "</svg>"
    return svgstring

def shape_svg(path):
    color = rgb_to_hex(path.color)
    return generate_stroke(path, path.z0, path.z2, color, path.w0, path.w2)

def rgb_to_hex(value):
    rgb = value * 255
    return "#%02x%02x%02x" % tuple(rgb.astype(int))