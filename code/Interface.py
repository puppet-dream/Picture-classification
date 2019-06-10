import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
import predict_Image as predict


class Surface(ttk.Frame):
    # class A(B):pass A继承B，不继承的话括号可写可不写，默认继承Object类
    pic_path = ""  # 定义变量pic_path是图片的路径
    viewhigh = 600  # 定义变量viewhigh为初始化图片高度
    viewwide = 600
    update_time = 0  # 更新的时间
    thread = None  # thread是一个模块
    thread_run = False  # thread_run是一个办法

    # 颜色的改变
    def __init__(self, win):
        # 定义类的时候，若是添加__init__方法，那么在创建类的实例的时候，实例会自动调用这个方法，一般用来对实例的属性进行初使化。
        # 定义完init()后，创建的每个实例都有自己的属性，也方便直接调用类中的函数。
        # kinter 共有三种几何布局管理器，分别是：pack布局，grid布局，place布局。 pack布局:
        # 学习的连接：https://blog.csdn.net/taotaohuoli/article/details/80333820?utm_source=blogxgwz8
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)  # 使用Frame办法增加一层容器 就是左边显示图片的容器
        frame_right1 = ttk.Frame(self)  # 使用Frame办法增加一层容器：用于存储并显示识别的数据
        frame_right2 = ttk.Frame(self)  # 使用Frame办法增加一层容器：用于让用户进行点击图片或者摄像头识别的操作
        win.title("车牌识别")  # 窗口的名称
        win.state("zoomed")  # state设置按钮组件状态,可选的有NORMAL、ACTIVE、 DISABLED。默认 NORMAL。zoomed是放大的意思
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        # fill设置组件是否水平或垂直方向填充
        # 值：（X,Y,BOTH,NONE）
        # fill=X（水平方向填充）
        # fill=Y(垂直方向填充)
        # fill=BOTH(水平和垂直)
        # fill=NONE（不填充）

        # expand 设置组件是否展开，当值为YSE是，side选项无效，组件显示在容器的中间位置，若fill为BOTH,填充容器的剩余空间。
        # expend=YES,expend=NO

        # ipadx,ipady  设置X方向（Y方向）内部间隙（与之并列的组件之间的间隔）  默认值0，非负整数，单位为像素
        frame_left.pack(side=LEFT, expand=1, fill=BOTH)
        # side 设置组件的对齐方式  值：（LEFT,TOP,RIGHT,BOTTOM）上下左右
        frame_right1.pack(side=TOP, expand=1, fill=tk.Y)
        frame_right2.pack(side=RIGHT, expand=0)
        # 创建label标签
        ttk.Label(frame_left, text='原图：').pack(anchor="nw")
        # ttk是tkinter里的一个模块，Label是ttk里的一个类 这个类有很多属性：https://www.cnblogs.com/mathpro/p/8052501.html
        # anchor  锚选项，当可用空间大于所需求空间时，决定控件放置于容器何处  N,E,S,W,NW,NE,SW,SE,CENTER(默认值)，八个方向以及中心

        # row,column row为行，column为例，设置组件放置于第几行第几例 取值为行，例的序号
        # sticky 设置组件在网格中的对齐方式 值：N、E、S、W、NW、NE、SW、SE、CENTER
        from_pic_ctl = ttk.Button(frame_right2, text="来自图片", width=20, command=self.from_pic)
        self.image_ctl = ttk.Label(frame_left)
        self.image_ctl.pack(anchor="nw")

        ttk.Label(frame_right1, text='识别结果：').grid(column=0, row=0, sticky=tk.W)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.grid(column=0, row=1, sticky=tk.W)

        from_pic_ctl.pack(anchor="se", pady="2")
        self.predictor = predict.CardPredictor()
        self.predictor.train_model()

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide != self.viewwide or high != self.viewhigh:
            im = im.resize((self.viewwide, self.viewhigh), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def from_pic(self):
        self.thread_run = False
        self.pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
        if self.pic_path:
            img = predict.image_read(self.pic_path)
            img_bgr = predict.imreadex(self.pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)
            r = self.predictor.predict(img)#获取识别结果
            self.r_ctl.configure(text=str(r))
            print(r)


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()
    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()  # 启动窗体
