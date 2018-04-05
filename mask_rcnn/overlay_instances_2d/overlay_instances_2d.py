import glob
from datetime import datetime
import numpy as np
from PIL import Image
from vispy import app, gloo
from vispy.gloo import context
from vispy.util import transforms
from matplotlib import pyplot as plt
from OpenGL import GL as gl

app.use_app('pyglet')

_vert_std = """
uniform mat4 u_mvp;
attribute vec2 a_pos;
attribute vec2 a_tex_coord;
varying vec2 v_tex_coord;

void main (void)
{
    gl_Position = u_mvp * vec4(a_pos, 0, 1);
    v_tex_coord = vec2(a_tex_coord.x, a_tex_coord.y);
}
"""

_frag_tex = """
uniform sampler2D u_tex;
varying vec2 v_tex_coord;

void main()
{
    vec4 c = texture2D(u_tex, v_tex_coord);
    if (c.a < 0.01)
        // do not draw and change depth buffer, 
        // when alpha is low
        discard;
    gl_FragColor = c;
}
"""


def _unit_square():
    return np.array([
        [0, 0], [1, 0], [0, 1],
        [0, 1], [1, 0], [1, 1],
    ], dtype=np.float32)


def _load_img(file):
    # flip view vertically: Image- to OpenGL coordinates
    img = Image.open(file).transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(img)


def generate_image_definition(bg_path, instance_paths, border_ratio=0.1,
                              min_scale=0.1, max_scale=1.0):
    with Image.open(bg_path) as img:
        w, h = img.size
    border = border_ratio * min(w, h)
    N = len(instance_paths)
    x = np.random.uniform(border, w - border, N)
    y = np.random.uniform(border, h - border, N)
    scales = np.random.uniform(min_scale, max_scale, N)
    rotations = np.random.uniform(0, 360, N)

    instances = []
    for i, instance_path in enumerate(instance_paths):
        with Image.open(instance_path) as img:
            iw, ih = img.size
        bg_instance_ratio = min(w, h) / min(iw, ih)
        instances.append({
            'path': instance_path, 'size': (iw, ih),
            'x': x[i], 'y': y[i],
            's': scales[i] * bg_instance_ratio,
            'r': rotations[i]
        })

    return {
        'background': {'path': bg_path, 'size': (w, h)},
        'instances': instances
    }


def generate_image(img_def):
    c = context.FakeCanvas()

    img = _load_img(img_def['background']['path'])
    h, w, _ = img.shape

    render_fbo = gloo.FrameBuffer(
        gloo.Texture2D(img),
        gloo.RenderBuffer((h, w))
    )

    program = gloo.Program(_vert_std, _frag_tex)
    program['a_pos'] = _unit_square()
    program['a_tex_coord'] = _unit_square()
    gloo.set_state(
        blend=True, blend_func=('one', 'one_minus_src_alpha'),
        depth_test=True, depth_func='always'
    )

    with render_fbo:
        gloo.set_viewport(0, 0, w, h)
        gloo.set_clear_depth(0)
        gloo.clear(depth=True, color=False)

        instances = img_def['instances']
        # The unsigned byte depth buffer extraction sets a limit of 255 instances.
        # Can be extracted as short if necessary.
        assert(len(instances) <= 255)
        for i, inst in enumerate(instances):
            img = _load_img(inst['path'])
            ih, iw, _ = img.shape
            x, y, s, r = (inst[k] for k in ['x', 'y', 's', 'r'])

            program['u_tex'] = gloo.Texture2D(img, interpolation='linear')
            program['u_mvp'] = \
                transforms.translate((-0.5, -0.5, 0)) @ \
                transforms.scale((s * iw, s * ih, 1)) @ \
                transforms.rotate(r, (0, 0, 1)) @ \
                transforms.translate((x, y, -i-1)) @ \
                transforms.ortho(0, w, 0, h, 0, 255)

            program.draw()

        rgb = render_fbo.read(alpha=False)
        depth = gl.glReadPixels(0, 0, w, h, gl.GL_DEPTH_COMPONENT, gl.GL_UNSIGNED_BYTE)
        if not isinstance(depth, np.ndarray):
            depth = np.frombuffer(depth, np.uint8)
        depth = np.flip(depth.reshape(h, w), axis=0)
        masks = np.empty((h, w, len(instances)), np.bool)
        for i in range(len(instances)):
            masks[:, :, i] = depth == i + 1

    return rgb, depth, masks


if __name__ == '__main__':
    instancefiles = glob.glob('./instances/**/*.png')

    img_def = generate_image_definition(
        bg_path='./bg.jpg',
        instance_paths=np.random.choice(instancefiles, 10),
        min_scale=0.4, max_scale=1.2
    )

    start = datetime.now()
    img, depth, masks = generate_image(img_def)
    print('Render took:', datetime.now() - start)

    rows, cols = 4, 2
    plt.subplot(rows, cols, 1)
    plt.imshow(img)
    plt.title('rgb')
    plt.axis('off')
    plt.subplot(rows, cols, 2)
    plt.imshow(depth)
    plt.title('depth')
    plt.axis('off')
    for i in range(min((rows - 1) * cols, depth.max())):
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(masks[:, :, i])
        plt.title('mask ' + str(i))
        plt.axis('off')
    plt.show()
