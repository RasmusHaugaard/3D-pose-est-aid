import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import sys
import os


class MaskCanvas:
    def __init__(self, img_list, out_dir, size=(640, 480), brush_size=25):
        self.img_list = img_list
        self.img_id = 0
        self.out_dir = out_dir

        self.size = size
        self.brush_size = brush_size

        self.img = None
        self.img_path = None
        self.mask = None
        self.mask_draw = None

        self.m = tk.Tk()
        self.m.title("Paint mask using ovals")
        self.c = tk.Canvas(
            self.m,
            width=640,
            height=480
        )
        self.c.pack(expand=tk.YES, fill=tk.BOTH)
        self.c.bind("<B1-Motion>", self.paint)
        self.m.bind("<space>", self.user_save)
        self.m.bind("<BackSpace>", self.user_reset)

        self.next()
        tk.mainloop()

    def next(self):
        # Clear canvas and mask
        self.c.delete("all")
        self.mask = Image.new("1", self.size, 1)
        self.mask_draw = ImageDraw.Draw(self.mask)

        if not self.img_id < len(self.img_list):
            self.m.destroy()
            return
        self.img_path = self.img_list[self.img_id]
        self.img_id += 1
        self.img = ImageTk.PhotoImage(Image.open(self.img_path))
        self.c.create_image(self.size[0]//2, self.size[1]//2, image=self.img)
        self.c.update()

    def paint(self, event):
        s = self.brush_size
        x1, y1 = (event.x - s), (event.y - s)
        x2, y2 = (event.x + s), (event.y + s)
        self.c.create_oval(x1, y1, x2, y2, fill="#000")
        self.mask_draw.ellipse((x1, y1, x2, y2), fill="#000")

    def user_save(self, _):
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        filename = os.path.basename(self.img_path)
        self.mask.save(self.out_dir + '/' + filename)
        self.next()

    def user_reset(self, _):
        self.img_id -= 1
        self.next()


def main():
    root_dir = sys.argv[1]
    rgb_dir = root_dir + '/rgb'
    assert os.path.isdir(rgb_dir)
    out_dir = root_dir + '/box_mask'
    rgb_list = [rgb_dir + '/' + rel_path for rel_path in os.listdir(rgb_dir)]

    MaskCanvas(rgb_list, out_dir)


if __name__ == '__main__':
    main()
