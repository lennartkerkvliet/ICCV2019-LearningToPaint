from .bezierpath import BezierPath
from .stroke_gen import generate_stroke


def render_svg(paths, width=128, corner_radius=True):
    svgstring = "<svg viewBox=\"0 0 {} {}\" xmlns=\"http://www.w3.org/2000/svg\">".format(width, width)
    for path in paths:
        svgstring += shape_svg(path, corner_radius)  
    svgstring += "</svg>"
    return svgstring

def shape_svg(path, corner_radius=True):
    color = rgb_to_hex(path.color)

    if corner_radius:
        return generate_stroke(path, path.z0, path.z2, color, path.w0, path.w2)
    else:
        return generate_stroke(path, path.z0, path.z2, color, max(path.w0, path.w2))

def rgb_to_hex(value):
    rgb = value * 255
    return "#%02x%02x%02x" % tuple(rgb.astype(int))