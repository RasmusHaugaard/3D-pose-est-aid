import numpy as np
from vispy import app
from vispy import gloo
from vispy.util import transforms
import cv2
import glob

app.use_app('pyqt5')

min_instances = 2
max_instances = 10

bg_img = cv2.imread('./dog.jpg', cv2.IMREAD_UNCHANGED)
h, w, _ = bg_img.shape
bg_tex = gloo.Texture2D(bg_img)

c = app.Canvas(keys='interactive', size=(w, h))

vertex = """
uniform mat4 u_mvp;
attribute vec2 a_pos;
attribute vec2 a_texcoord;
varying vec2 v_texcoord;

void main (void)
{
    gl_Position = u_mvp * vec4(a_pos, 0, 1);
    v_texcoord = vec2(a_texcoord.x, 1.0 - a_texcoord.y);
}
"""

fragment = """
uniform sampler2D u_tex;
varying vec2 v_texcoord;

vec4 c;

void main()
{
    c = texture2D(u_tex, v_texcoord);
    gl_FragColor = vec4(c.b, c.g, c.r, c.a);
}
"""

program = gloo.Program(vertex, fragment)

program['a_pos'] = np.array([
    [0, 0], [1, 0], [0, 1],
    [0, 1], [1, 0], [1, 1],
], dtype=np.float32)

program['a_texcoord'] = np.array([
    [0, 0], [1, 0], [0, 1],
    [0, 1], [1, 0], [1, 1],
], dtype=np.float32)

cupImages = glob.glob('./cups/*.png')

gloo.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))


@c.connect
def on_resize(event):
    gloo.set_viewport(0, 0, *event.size)


@c.connect
def on_draw(event):
    projection = transforms.ortho(0, w, 0, h, -1, 1)

    mvp = np.eye(4, dtype=np.float32)
    mvp = mvp.dot(transforms.scale((w, h, 1)))
    mvp = mvp.dot(projection)
    program['u_mvp'] = mvp
    program['u_tex'] = bg_tex
    program.draw('triangles')

    cup_count = np.random.randint(min_instances, max_instances + 1)
    for _ in range(cup_count):
        x = np.random.uniform(0, w)
        y = np.random.uniform(0, h)
        r = np.random.uniform(0, 360)
        s = np.random.uniform(0.1, 1.5)
        img = cv2.imread(np.random.choice(cupImages), cv2.IMREAD_UNCHANGED)
        ih, iw, _ = img.shape
        program['u_tex'] = gloo.Texture2D(img)

        mvp = np.eye(4, dtype=np.float32)
        mvp = mvp.dot(transforms.translate((-0.5, -0.5, 0)))
        mvp = mvp.dot(transforms.scale((s * iw, s * ih, s)))
        mvp = mvp.dot(transforms.rotate(r * 360, (0, 0, 1)))
        mvp = mvp.dot(transforms.translate((x, y, 0)))
        mvp = mvp.dot(projection)
        program['u_mvp'] = mvp

        program.draw('triangles')


c.show()
app.run()
