import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PyQt6.QtWidgets   import QDialog, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QProgressBar
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from scipy import ndimage
import pickle as pkl
import Tools



class FilteringWindow(QDialog):
    def __init__(self, disk_name, image, threshold, x_min, x_max, parent=None):
        super(FilteringWindow, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
        self.resize(1000, 600)
        # self.setFixedSize(1500, 900)
        self.nb_steps = 10000
        self.disk_name = disk_name
        self.image = image
        self.Lim_Radius = int(len(image)/2 - 1)
        self.threshold = threshold
        self.x_min, self.x_max = x_min, x_max
        self.R_max = int(x_max - len(self.image)/2)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Filtering Pixel position Data Window')

        self.Filtering_Fig, self.Filtering_ax = plt.subplots(1, 1, num="Filtering Data")
        self.Filtering_Canvas = FigureCanvas(self.Filtering_Fig)
        self.Filtering_Canvas.setParent(self)
        

        close_button = QPushButton('Save Data - Close', self)
        close_button.clicked.connect(self.Continue)
        close_button.setFixedHeight(40)
        close_button.setFixedWidth(300)
        
        self.Filtering_Canvas.mpl_connect('button_press_event', self.on_press)
        self.Filtering_Canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.Filtering_Canvas.mpl_connect('button_release_event', self.on_release)

        # Ajoutez un slider pour le zoom
        self.Gaussian_Label  = QLabel("Gaussian Filter Parameter : ", self)
        self.Gaussian_slider = QSlider(Qt.Orientation.Horizontal)
        self.Gaussian_slider.setMinimum(1)
        self.Gaussian_slider.setMaximum(1000)
        self.Gaussian_slider.setValue(1)
        self.gaussian_value = 0.01
        self.Gaussian_value_Label = QLabel(str(self.gaussian_value))
        self.Gaussian_slider.setSingleStep(1)
        self.Gaussian_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.Gaussian_Label.setFixedWidth(200)
        self.Gaussian_value_Label.setFixedWidth(100)
        self.Gaussian_slider.setFixedWidth(300)

        self.Smooth_Label  = QLabel("Smooth Profile Parameter : ", self)
        self.Smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.Smooth_slider.setMinimum(1)
        self.Smooth_slider.setMaximum(10)
        self.Smooth_slider.setValue(1)
        self.smooth_value = 1
        self.Smooth_value_Label = QLabel(str(self.smooth_value))
        self.Smooth_slider.setSingleStep(1)
        self.Smooth_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.Smooth_Label.setFixedWidth(200)
        self.Smooth_value_Label.setFixedWidth(100)
        self.Smooth_slider.setFixedWidth(300)

        self.Distance_Label  = QLabel("Distance Peaks Parameter : ", self)
        self.Distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.Distance_slider.setMinimum(100)
        self.Distance_slider.setMaximum(10000)
        self.Distance_slider.setValue(100)
        self.Distance_value = 1
        self.Distance_value_Label = QLabel(str(self.Distance_value))
        self.Distance_slider.setSingleStep(1)
        self.Distance_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.Distance_Label.setFixedWidth(200)
        self.Distance_value_Label.setFixedWidth(100)
        self.Distance_slider.setFixedWidth(300)

        self.Prominence_Label  = QLabel("Prominence Peaks Parameter : ", self)
        self.Prominence_slider = QSlider(Qt.Orientation.Horizontal)
        self.Prominence_slider.setMinimum(-500)
        self.Prominence_slider.setMaximum(500)
        self.Prominence_slider.setValue(0)
        self.Prominence_value = 10**(0/100)
        self.Prominence_value_Label = QLabel(str(self.Prominence_value))
        self.Prominence_slider.setSingleStep(1)
        self.Prominence_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.Prominence_Label.setFixedWidth(200)
        self.Prominence_value_Label.setFixedWidth(100)
        self.Prominence_slider.setFixedWidth(300)

        self.Width_Label  = QLabel("Width Peaks Parameter : ", self)
        self.Width_slider = QSlider(Qt.Orientation.Horizontal)
        self.Width_slider.setMinimum(1)
        self.Width_slider.setMaximum(10000)
        self.Width_slider.setValue(100)
        self.Width_value = 0.1
        self.Width_value_Label = QLabel(str(self.Width_value))
        self.Width_slider.setSingleStep(1)
        self.Width_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.Width_Label.setFixedWidth(200)
        self.Width_value_Label.setFixedWidth(100)
        self.Width_slider.setFixedWidth(300)

        self.HighPass_Label  = QLabel("High Pass Parameter : ", self)
        self.HighPass_slider = QSlider(Qt.Orientation.Horizontal)
        self.HighPass_slider.setMinimum(10)
        self.HighPass_slider.setMaximum(1000)
        self.HighPass_slider.setValue(10)
        self.HighPass_value = 21
        self.HighPass_value_Label = QLabel(str(self.HighPass_value))
        self.HighPass_slider.setSingleStep(1)
        self.HighPass_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.HighPass_Label.setFixedWidth(200)
        self.HighPass_value_Label.setFixedWidth(100)
        self.HighPass_slider.setFixedWidth(300)

        self.MinCutRad_Label  = QLabel("Min Radius Cut : ", self)
        self.MinCutRad_slider = QSlider(Qt.Orientation.Horizontal)
        self.MinCutRad_slider.setMinimum(1)
        self.MinCutRad_slider.setMaximum(self.Lim_Radius - 1)
        self.MinCutRad_slider.setValue(2)
        self.MinCutRad_value = 2
        self.MinCutRad_value_Label = QLabel(str(self.MinCutRad_value))
        self.MinCutRad_slider.setSingleStep(1)
        self.MinCutRad_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.MinCutRad_Label.setFixedWidth(200)
        self.MinCutRad_value_Label.setFixedWidth(100)
        self.MinCutRad_slider.setFixedWidth(300)

        self.MaxCutRad_Label  = QLabel("Max Radius Cut : ", self)
        self.MaxCutRad_slider = QSlider(Qt.Orientation.Horizontal)
        self.MaxCutRad_slider.setMinimum(1)
        self.MaxCutRad_slider.setMaximum(self.Lim_Radius)
        self.MaxCutRad_slider.setValue(100)
        self.MaxCutRad_value = 100
        self.MaxCutRad_value_Label = QLabel(str(self.MaxCutRad_value))
        self.MaxCutRad_slider.setSingleStep(1)
        self.MaxCutRad_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.MaxCutRad_Label.setFixedWidth(200)
        self.MaxCutRad_value_Label.setFixedWidth(100)
        self.MaxCutRad_slider.setFixedWidth(300)

        self.Azimuthal_Method    = QPushButton("Azimutal Cut", self)
        self.Diagonal_Method     = QPushButton("Diagonal Cut", self)
        self.Antidiagonal_Method = QPushButton("Antidiagonal Cut", self)
        self.Horizontal_Method   = QPushButton("Horizontal Cut", self)
        self.Vertical_Method     = QPushButton("Vertical Cut", self)
        self.Method_Value = "Azimuthal"
        self.Azimuthal_Method.setFixedWidth(300)
        self.Diagonal_Method.setFixedWidth(145)
        self.Antidiagonal_Method.setFixedWidth(145)
        self.Horizontal_Method.setFixedWidth(145)
        self.Vertical_Method.setFixedWidth(145)

        self.Gaussian_slider.valueChanged.connect(lambda   value : self.Change_Fit(value, Type='Gaussian'))
        self.Smooth_slider.valueChanged.connect(lambda     value : self.Change_Fit(value, Type='Smooth'))
        self.Distance_slider.valueChanged.connect(lambda   value : self.Change_Fit(value, Type='Distance'))
        self.Prominence_slider.valueChanged.connect(lambda value : self.Change_Fit(value, Type='Prominence'))
        self.Width_slider.valueChanged.connect(lambda      value : self.Change_Fit(value, Type='Width'))
        self.HighPass_slider.valueChanged.connect(lambda   value : self.Change_Fit(value, Type='HighPass'))
        self.MinCutRad_slider.valueChanged.connect(lambda  value : self.Change_Fit(value, Type='MinCutRad'))
        self.MaxCutRad_slider.valueChanged.connect(lambda  value : self.Change_Fit(value, Type='MaxCutRad'))
        self.Azimuthal_Method.clicked.connect(lambda       value : self.Change_Fit(0, Type='Azimuthal'))
        self.Diagonal_Method.clicked.connect(lambda        value : self.Change_Fit(0, Type='Diagonal'))
        self.Antidiagonal_Method.clicked.connect(lambda    value : self.Change_Fit(0, Type='Antidiagonal'))
        self.Horizontal_Method.clicked.connect(lambda      value : self.Change_Fit(0, Type='Horizontal'))
        self.Vertical_Method.clicked.connect(lambda        value : self.Change_Fit(0, Type='Vertical'))
        self.Gaussian_Label.setFixedHeight(10)
        self.Gaussian_value_Label.setFixedHeight(10)
        self.Smooth_Label.setFixedHeight(10)
        self.Smooth_value_Label.setFixedHeight(10)
        self.Distance_Label.setFixedHeight(10)
        self.Distance_value_Label.setFixedHeight(10)
        self.Prominence_Label.setFixedHeight(10)
        self.Prominence_value_Label.setFixedHeight(10)
        self.Width_Label.setFixedHeight(10)
        self.Width_value_Label.setFixedHeight(10)
        self.HighPass_Label.setFixedHeight(10)
        self.HighPass_value_Label.setFixedHeight(10)
        self.MinCutRad_Label.setFixedHeight(10)
        self.MinCutRad_value_Label.setFixedHeight(10)
        self.MaxCutRad_Label.setFixedHeight(10)
        self.MaxCutRad_value_Label.setFixedHeight(10)

        self.Azimuthal_Method.setFixedHeight(25)
        self.Diagonal_Method.setFixedHeight(25)
        self.Antidiagonal_Method.setFixedHeight(25)
        self.Vertical_Method.setFixedHeight(25)
        self.Horizontal_Method.setFixedHeight(25)

        self.progress = QProgressBar(self)
        self.progress.setMaximum(self.nb_steps)
        self.progress.setFixedHeight(10)
        self.progress.setFixedWidth(300)

        DiagonalButton = QHBoxLayout()
        DiagonalButton.addWidget(self.Diagonal_Method)
        DiagonalButton.addWidget(self.Antidiagonal_Method)

        VertHorizButton = QHBoxLayout()
        VertHorizButton.addWidget(self.Vertical_Method)
        VertHorizButton.addWidget(self.Horizontal_Method)

        MethodButton = QVBoxLayout()
        MethodButton.addWidget(self.Azimuthal_Method)
        MethodButton.addLayout(DiagonalButton)
        MethodButton.addLayout(VertHorizButton)

        Filtering_Layout = QVBoxLayout()
        
        GaussianLayout = QHBoxLayout()
        GaussianLayout.addWidget(self.Gaussian_Label)
        GaussianLayout.addWidget(self.Gaussian_value_Label)
        Filtering_Layout.addLayout(GaussianLayout)
        Filtering_Layout.addWidget(self.Gaussian_slider)

        SmoothLayout = QHBoxLayout()
        SmoothLayout.addWidget(self.Smooth_Label)
        SmoothLayout.addWidget(self.Smooth_value_Label)
        Filtering_Layout.addLayout(SmoothLayout)
        Filtering_Layout.addWidget(self.Smooth_slider)

        DistanceLayout = QHBoxLayout()
        DistanceLayout.addWidget(self.Distance_Label)
        DistanceLayout.addWidget(self.Distance_value_Label)
        Filtering_Layout.addLayout(DistanceLayout)
        Filtering_Layout.addWidget(self.Distance_slider)

        ProminenceLayout = QHBoxLayout()
        ProminenceLayout.addWidget(self.Prominence_Label)
        ProminenceLayout.addWidget(self.Prominence_value_Label)
        Filtering_Layout.addLayout(ProminenceLayout)
        Filtering_Layout.addWidget(self.Prominence_slider)

        WidthLayout = QHBoxLayout()
        WidthLayout.addWidget(self.Width_Label)
        WidthLayout.addWidget(self.Width_value_Label)
        Filtering_Layout.addLayout(WidthLayout)
        Filtering_Layout.addWidget(self.Width_slider)

        HighPassLayout = QHBoxLayout()
        HighPassLayout.addWidget(self.HighPass_Label)
        HighPassLayout.addWidget(self.HighPass_value_Label)
        Filtering_Layout.addLayout(HighPassLayout)
        Filtering_Layout.addWidget(self.HighPass_slider)

        MinCutLayout = QHBoxLayout()
        MinCutLayout.addWidget(self.MinCutRad_Label)
        MinCutLayout.addWidget(self.MinCutRad_value_Label)
        Filtering_Layout.addLayout(MinCutLayout)
        Filtering_Layout.addWidget(self.MinCutRad_slider)

        MaxCutLayout = QHBoxLayout()
        MaxCutLayout.addWidget(self.MaxCutRad_Label)
        MaxCutLayout.addWidget(self.MaxCutRad_value_Label)
        Filtering_Layout.addLayout(MaxCutLayout)
        Filtering_Layout.addWidget(self.MaxCutRad_slider)

        Filtering_Layout.addLayout(MethodButton)
        Filtering_Layout.addWidget(close_button)
        Filtering_Layout.addWidget(self.progress)

        Figure_Layout = QVBoxLayout()

        Figure_Layout.addWidget(self.Filtering_Canvas)
        # Figure_Layout.addWidget(close_button)

        All_Layout = QHBoxLayout()
        All_Layout.addLayout(Filtering_Layout)
        All_Layout.addLayout(Figure_Layout)
        self.setLayout(All_Layout)

        # Créer une figure et un canevas
        self.rect_start = None
        self.rect_preview = None
        X, Y = Tools.Max_pixel(self.image, R_max=self.R_max, gaussian_filter=self.gaussian_value, smooth_filter=self.smooth_value)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Filtering_ax.set_facecolor('k')
        self.displayed_image = self.Filtering_ax.imshow(self.image, cmap="gnuplot", norm=colors.SymLogNorm(linthresh=self.threshold))
        self.Filtering_ax.set_xlim(self.x_min, self.x_max)
        self.Filtering_ax.set_ylim(self.x_min, self.x_max)
        self.Data = self.Filtering_ax.scatter(self.Y, self.X, edgecolor='k', color='cyan', s=5)
        self.Change_Fit(1, 'None')

        self.X_rect_press   = []
        self.Y_rect_press   = []
        self.X_rect_release = []
        self.Y_rect_release = []

    def Change_Fit(self, value, Type):
        if Type=='Gaussian':
            self.gaussian_value = value /100
            self.Gaussian_value_Label.setText(str(self.gaussian_value))
            self.displayed_image.remove()
            self.displayed_image = self.Filtering_ax.imshow(ndimage.gaussian_filter(self.image , sigma = self.gaussian_value), cmap="gnuplot", norm=colors.SymLogNorm(linthresh=self.threshold))

        elif Type=='Smooth':
            self.smooth_value = value
            self.Smooth_value_Label.setText(str(self.smooth_value))
        elif Type=='Distance':
            if value == 100:
                self.Distance_value = None
            else :
                self.Distance_value = value/100
            self.Distance_value_Label.setText(str(self.Distance_value))
        elif Type=='Prominence':
            self.Prominence_value = 10**(value/100)
            self.Prominence_value_Label.setText(str(self.Prominence_value))
        elif Type=='Width':
            self.Width_value = value/1000
            self.Width_value_Label.setText(str(self.Width_value))
        elif Type=='HighPass':
            self.HighPass_value = value*2 + 1
            self.HighPass_value_Label.setText(str(self.HighPass_value))
        elif Type=='MinCutRad':
            self.MinCutRad_value = value
            self.MinCutRad_value_Label.setText(str(self.MinCutRad_value))
            if self.MinCutRad_value >= self.MaxCutRad_value:
                self.MaxCutRad_value = value + 1
                self.MaxCutRad_slider.setValue(value + 1)
                self.MaxCutRad_value_Label.setText(str(self.MaxCutRad_value))
        elif Type=='MaxCutRad':
            self.MaxCutRad_value = value
            self.MaxCutRad_value_Label.setText(str(self.MaxCutRad_value))
            if self.MaxCutRad_value <= self.MinCutRad_value:
                self.MinCutRad_value = value - 1
                self.MinCutRad_slider.setValue(value-1)
                self.MinCutRad_value_Label.setText(str(self.MinCutRad_value))
            
        elif Type=="Azimuthal" or Type=='Diagonal' or Type=='Antidiagonal' or Type=='Vertical' or Type=='Horizontal':
            self.Method_Value = Type
        self.Data.remove()
        X, Y = Tools.Max_pixel(self.image, R_max = self.R_max,  gaussian_filter = self.gaussian_value, 
                                                                smooth_filter   = self.smooth_value, 
                                                                prominence      = self.Prominence_value, 
                                                                distance        = self.Distance_value, 
                                                                width           = self.Width_value, 
                                                                threshold       = None,
                                                                HighPass        = self.HighPass_value,
                                                                Mincut_Radius   = self.MinCutRad_value,
                                                                Maxcut_Radius   = self.MaxCutRad_value,
                                                                method          = self.Method_Value)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Data = self.Filtering_ax.scatter(self.Y, self.X, edgecolor='k', color='cyan', s=5)
        self.Filtering_Canvas.draw()

    def on_press(self, event):
        if event.inaxes == self.Filtering_ax:
            self.x_press, self.y_press = event.xdata, event.ydata
            self.X_rect_press.append(self.x_press)
            self.Y_rect_press.append(self.y_press)
            self.rect_start = (self.x_press, self.y_press)
            self.rect_preview = Rectangle((self.x_press, self.y_press), 0, 0, edgecolor='navy', facecolor='blue', alpha=0.2)
            self.Filtering_ax.add_patch(self.rect_preview)
            self.Filtering_Canvas.draw()

    def on_motion(self, event):
        if self.rect_start is not None and event.inaxes == self.Filtering_ax:
            x, y = event.xdata, event.ydata
            width = x - self.rect_start[0]
            height = y - self.rect_start[1]
            self.rect_preview.set_width(width)
            self.rect_preview.set_height(height)
            self.Filtering_Canvas.draw()

    def on_release(self, event):
        if self.rect_start is not None and event.inaxes == self.Filtering_ax:
            self.x_release, self.y_release = event.xdata, event.ydata
            self.X_rect_release.append(self.x_release)
            self.Y_rect_release.append(self.y_release)
            y1 = min(self.x_press, self.x_release)
            y2 = max(self.x_press, self.x_release)
            x1 = min(self.y_press, self.y_release)
            x2 = max(self.y_press, self.y_release)

            b = np.where(np.logical_or(np.logical_or(self.X<x1, self.X>x2), np.logical_or(self.Y<y1, self.Y>y2)))
            self.X = self.X[b]
            self.Y = self.Y[b]
            self.Data.remove()
            self.Data = self.Filtering_ax.scatter(self.Y, self.X, edgecolor='k', color='cyan', s=5)
            self.rect_preview.remove()


        self.rect_start = None
        self.rect_preview = None
        self.Filtering_Canvas.draw()

    def Continue(self):
        Fit_Name = 'Fitting_' + self.disk_name +'.pkl'
        x0 = y0 = int(len(self.image)/2)

        coeffs = Tools.fit_ellipse(np.array(self.X), np.array(self.Y))
        Xc, Yc, a, b, e, PA_LSQE = Tools.cart_to_pol(coeffs)

        incl     = np.arccos(b/a)
        PA       = Tools.My_PositionAngle(x0, y0, Yc, Xc, a, b, e, PA_LSQE)
        X_e, Y_e = Tools.get_ellipse_pts((Xc, Yc, a, b, e, PA))
        H        = Tools.Height_Compute(X_e, Y_e, x0, y0)/np.sin(incl)
        R        = a
        Aspect   = H/R
        Ellipse_points = [Xc, Yc, X_e, Y_e, a, b, e]

        Point_Offset = []
        for idx in range(len(self.X)):
            idx_dist     = np.argmin(np.sqrt((X_e - self.X[idx])**2 + (Y_e - self.Y[idx])**2))
            ellipse_dist = np.sqrt((Xc - X_e[idx_dist])**2 + (Yc - Y_e[idx_dist])**2)
            point_dist   = np.sqrt((Xc - self.X[idx])**2        + (Yc - self.Y[idx])**2)
            dist         = ellipse_dist - point_dist
            Point_Offset.append(dist)
        
        Inclinations    = []
        PositionAngles  = []
        Heights         = []
        Mid_Radius      = []
        for step in range(self.nb_steps):
            self.progress.setValue(step + 1)
            nb_points = 100
            Radius       = np.random.normal(R, np.std(Point_Offset), nb_points)
            Phi          = np.random.uniform(0, 2*np.pi, nb_points)
            Xrand, Yrand = Tools.Random_Ellipse(Radius, Phi, x0, y0, Xc, Yc, incl, R, H, Aspect, 1, PA)
            coeffs = Tools.fit_ellipse(np.array(Xrand), np.array(Yrand))
            Xc, Yc, a, b, e, PA_LSQE = Tools.cart_to_pol(coeffs)
            Inclinations.append(np.arccos(b/a))
            PositionAngles.append(Tools.My_PositionAngle(x0, y0, Yc, Xc, a, b, e, PA_LSQE))
            # X_e, Y_e      = Tools.get_ellipse_pts((Xc, Yc, a, b, e, PositionAngle))
            Heights.append(np.sqrt((x0-Xc)**2 + (y0-Yc)**2)/np.sin(np.arccos(b/a)))
            Mid_Radius.append(a)

        D_incl   = np.std(Inclinations)
        D_PA     = np.std(PositionAngles)
        D_H      = np.std(Heights)
        D_R      = np.std(Mid_Radius)
        D_Chi    = 0
        D_Aspect = np.std(np.array(Heights)/np.array(Mid_Radius))
        Data_to_Save    = {"params"     : [incl, R, H, Aspect, 1, PA], 
                        "Err"        : [D_incl, D_R, D_H, D_Aspect, D_Chi, D_PA], 
                        'Points'     : [self.X, self.Y], 
                        "Ellipse"    : Ellipse_points}
        GUI_Folder, Fitting_Folder, SPF_Folder = Tools.Init_Folders()
        with open(f"{Fitting_Folder}/{Fit_Name}", 'wb') as fichier:
            pkl.dump(Data_to_Save, fichier)

        # Tools.Fitting_Ring(x0, y0, self.X, self.Y, 100, Fit_Name, Tsigma=3)
        # Tools.New_Fitting_Ring(x0, y0, self.X, self.Y, 100, Fit_Name, np.shape(self.image), Tsigma=3)
        self.Filtering_Fig.clf()
        self.Filtering_Canvas.flush_events()
        plt.close(self.Filtering_Fig)
        self.accept()
