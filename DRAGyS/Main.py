import sys
import os
import pathlib
current_folder = pathlib.Path(__file__).parent
print(current_folder)
sys.path.append(current_folder)
import numpy as np
import time
import pickle as pkl
from scipy.interpolate import griddata
from PyQt5.QtWidgets import QTextEdit, QApplication, QWidget, QSlider, QCheckBox, QSpinBox, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QFrame, QProgressBar
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib import colors

from astropy.io import fits
import Tools as Tools
import SPF_Window as SPF
from   Filter_Pixel import FilteringWindow


class FileExplorerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.normalization      = False
        self.LogScale           = False
        self.InputParams        = False
        self.InputStar          = False 
        self.Image_Displayed    = False
        self.Ellipse_Extraction = False 
        self.ellipse_in         = False  
        self.img_name = ''
        
        self.Fitting = None

        self.unit = 'Arcsec'
        
        self.List_Buttons = []
        self.List_Frame   = []

        self.GUI_Folder, self.Fitting_Folder, self.SPF_Folder = Tools.Init_Folders()

        # Bouton pour parcourir les fichiers
        self.browse_button = QPushButton('Browse Files', self)
        self.browse_button.clicked.connect(self.browse_files)

        self.file_label = QLabel('No file selected', self)
        self.file_label.setFixedHeight(15)
        self.file_label.setStyleSheet('font-size: 14px;')
        self.fit_file = QLabel("Is Fitting already done ? None", self)
        self.fit_file.setFixedHeight(15)

        # Fitting buttons
        self.Display_Fit_button = QPushButton('Show Fitting', self)
        self.Display_Fit_button.clicked.connect(self.Show_Fitting)
        self.Display_Fit_button.setEnabled(False)
        self.Compute_Fit_button = QPushButton('Compute Fitting', self)
        self.Compute_Fit_button.clicked.connect(self.Launch_Filtering_Data)
        self.Compute_Fit_button.setEnabled(False)

        self.CheckInputParams = QCheckBox('Input Parameters', self)
        self.CheckInputParams.setStyleSheet('color: black; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
        self.CheckInputParams.stateChanged.connect(self.on_check_Input_Parameters)

        self.Inclination_text   = QLabel('i   = ', self)
        self.PositionAngle_text = QLabel('PA  = ', self)
        self.Scale_Height_text  = QLabel('h/r = ', self)
        self.Height_text  = QLabel('h = ', self)
        self.Radius_text  = QLabel('r = ', self)
        self.Alpha_text  = QLabel('alpha = ', self)

        self.Err_Inclination_text   = QLabel('° \u00B1 ', self)
        self.Err_PositionAngle_text = QLabel('° \u00B1  ', self)
        self.Err_Scale_Height_text  = QLabel(' \u00B1 ', self)
        self.Err_Height_text  = QLabel(' \u00B1 ', self)
        self.Err_Radius_text  = QLabel(' \u00B1 ', self)
        self.Err_Alpha_text  = QLabel(' \u00B1 ', self)

        self.fit_file.setStyleSheet('font-size: 14px; color: red;')
        self.Inclination_text.setStyleSheet('font-size: 14px; color: grey;')
        self.PositionAngle_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Scale_Height_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Height_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Radius_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Alpha_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Err_Inclination_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Err_PositionAngle_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Err_Scale_Height_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Err_Height_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Err_Radius_text.setStyleSheet('font-size: 14px; color: grey;')
        self.Err_Alpha_text.setStyleSheet('font-size: 14px; color: grey;')

        self.InclinationLine   = QLineEdit(self)
        self.PositionAngleLine = QLineEdit(self)
        self.AspectRatioLine   = QLineEdit(self)
        self.HeightLine   = QLineEdit(self)
        self.RadiusLine   = QLineEdit(self)
        self.PowerLawAlphaLine = QLineEdit(self)

        self.ErrInclinationLine   = QLineEdit(self)
        self.ErrPositionAngleLine = QLineEdit(self)
        self.ErrAspectRatioLine   = QLineEdit(self)
        self.ErrHeightLine   = QLineEdit(self)
        self.ErrRadiusLine   = QLineEdit(self)
        self.ErrPowerLawAlphaLine = QLineEdit(self)

        self.InclinationLine.setEnabled(False)
        self.PositionAngleLine.setEnabled(False)
        self.AspectRatioLine.setEnabled(False)
        self.HeightLine.setEnabled(False)
        self.RadiusLine.setEnabled(False)
        self.PowerLawAlphaLine.setEnabled(False)

        self.ErrInclinationLine.setEnabled(False)
        self.ErrPositionAngleLine.setEnabled(False)
        self.ErrHeightLine.setEnabled(False)
        self.ErrRadiusLine.setEnabled(False)
        self.ErrAspectRatioLine.setEnabled(False)
        self.ErrPowerLawAlphaLine.setEnabled(False)

        self.RadiusLine.textChanged.connect(self.update_params)
        self.ErrRadiusLine.textChanged.connect(self.update_params)
        self.HeightLine.textChanged.connect(self.update_params)
        self.ErrHeightLine.textChanged.connect(self.update_params)
        self.PowerLawAlphaLine.textChanged.connect(self.update_params)

        self.InclinationLine.textChanged.connect(self.Extraction_Zone)
        self.PositionAngleLine.textChanged.connect(self.Extraction_Zone)
        self.HeightLine.textChanged.connect(self.Extraction_Zone)
        self.RadiusLine.textChanged.connect(self.Extraction_Zone)
        self.PowerLawAlphaLine.textChanged.connect(self.Extraction_Zone)
        

        self.CheckStar = QCheckBox('Non Centered Star', self)
        self.CheckStar.setStyleSheet('color: grey; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
        self.X_StarPositionLabel = QLabel("(x = ", self)
        self.X_StarPosition = QLineEdit(self)
        self.Y_StarPositionLabel = QLabel(", y = ", self)
        self.Y_StarPosition = QLineEdit(self)
        self.StarPositionLabel = QLabel(")", self)
        self.X_StarPositionLabel.setStyleSheet('font-size: 14px; color: grey;')
        self.Y_StarPositionLabel.setStyleSheet('font-size: 14px; color: grey;')
        self.StarPositionLabel.setStyleSheet('font-size: 14px; color: grey;')
        self.CheckStar.stateChanged.connect(self.on_check_Star_Position)
        self.CheckStar.setChecked(False)
        self.CheckStar.setEnabled(False)
        self.X_StarPosition.setEnabled(False)
        self.Y_StarPosition.setEnabled(False)
        # Image View buttons
        self.img_0_button = QPushButton('Image 1', self)
        self.img_1_button = QPushButton('Image 2', self)
        self.img_2_button = QPushButton('Image 3', self)
        self.img_3_button = QPushButton('Image 4', self)
        self.img_4_button = QPushButton('Image 5', self)
        self.img_5_button = QPushButton('Image 6', self)
        self.HeaderButton = QPushButton('Header',  self)

        self.img_0_button.clicked.connect(lambda: self.Change_View(self.display_0))
        self.img_1_button.clicked.connect(lambda: self.Change_View(self.display_1))
        self.img_2_button.clicked.connect(lambda: self.Change_View(self.display_2))
        self.img_3_button.clicked.connect(lambda: self.Change_View(self.display_3))
        self.img_4_button.clicked.connect(lambda: self.Change_View(self.display_4))
        self.img_5_button.clicked.connect(lambda: self.Change_View(self.display_5))
        self.HeaderButton.clicked.connect(self.Open_Header)

       
        self.img_0_button.setEnabled(False)
        self.img_1_button.setEnabled(False)
        self.img_2_button.setEnabled(False)
        self.img_3_button.setEnabled(False)
        self.img_4_button.setEnabled(False)
        self.img_5_button.setEnabled(False)
        self.HeaderButton.setEnabled(False)

        self.HeaderButton.setFixedHeight(80)

        # Parameters buttons
        self.pixelscale_label = QLabel('Pixelscale ("/pix) : ', self)
        self.distance_label   = QLabel('Distance (pc) : ', self)
        self.R_in_label       = QLabel('R_in (au) : ', self)
        self.R_out_label      = QLabel('R_out (au) : ', self)
        self.n_bin_label      = QLabel('nb bins : ', self)

        self.pixelscale_label.setStyleSheet('font-size: 14px;')
        self.distance_label.setStyleSheet('font-size: 14px;')
        self.R_in_label.setStyleSheet('font-size: 14px;')
        self.R_out_label.setStyleSheet('font-size: 14px;')
        self.n_bin_label.setStyleSheet('font-size: 14px;')

        self.auUnitButton     = QPushButton('au', self)
        self.PixelUnitButton  = QPushButton('Pixel', self)
        self.ArcsecUnitButton = QPushButton('Arcsec', self)

        # self.auUnitButton.clicked.connect(lambda: self.ChangeUnit('au'))
        # self.PixelUnitButton.clicked.connect(lambda: self.ChangeUnit('Pixel'))
        # self.ArcsecUnitButton.clicked.connect(lambda: self.ChangeUnit('Arcsec'))
        self.auUnitButton.setEnabled(False)
        self.ArcsecUnitButton.setEnabled(False)
        self.PixelUnitButton.setEnabled(False)

        self.pixelscale_entry = QLineEdit(self)
        self.distance_entry   = QLineEdit(self)
        self.R_in_entry       = QSpinBox(self)
        self.R_out_entry      = QSpinBox(self)
        self.R_in_entry.setMinimum(0)
        self.R_in_entry.setMaximum(1000)
        self.R_out_entry.setMinimum(0)
        self.R_out_entry.setMaximum(1000)
        self.R_adjustement    = QCheckBox('See Adjustment', self)
        self.R_in_entry.valueChanged.connect(self.Extraction_Zone)
        self.R_out_entry.valueChanged.connect(self.Extraction_Zone)
        self.R_in_entry.lineEdit().returnPressed.connect(self.Extraction_Zone)
        self.R_out_entry.lineEdit().returnPressed.connect(self.Extraction_Zone)
        self.R_adjustement.stateChanged.connect(self.Ring_Adjust_2)
        self.nb_bin_entry     = QLineEdit(self)

        self.pixelscale_entry.setStyleSheet('color: black; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
        self.distance_entry.setStyleSheet('color: black; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
        self.R_in_entry.setStyleSheet('color: black; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
        self.R_out_entry.setStyleSheet('color: black; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
        self.nb_bin_entry.setStyleSheet('color: black; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')


        self.pixelscale_entry.setEnabled(False)
        self.distance_entry.setEnabled(False)
        self.R_in_entry.setEnabled(False)
        self.R_out_entry.setEnabled(False)
        self.R_adjustement.setEnabled(False)
        self.nb_bin_entry.setEnabled(False)
        self.pixelscale_entry.setEnabled(False)

        self.pixelscale_entry.setFixedWidth(40)
        self.distance_entry.setFixedWidth(40)
        self.nb_bin_entry.setFixedWidth(40)

        # Zoom button
        self.ZoomLabel  = QLabel("Zoom : ", self)
        self.ZoomLabel.setStyleSheet('color: black; font-size: 14px;')
        self.ZoomLabel.setFixedHeight(15)
        self.ZoomSlider = QSlider(Qt.Horizontal)
        self.ZoomSlider.setMinimum(100)
        self.ZoomSlider.setMaximum(2000)
        self.ZoomSlider.setValue(1)
        self.ZoomSlider.setSingleStep(1)
        self.ZoomSlider.valueChanged.connect(self.Zoom_Slider_Update)

        # PhFs button
        self.Compute_PhF_button = QPushButton('Compute \n Phase functions', self)
        self.Show_disk_PhF_button = QPushButton('Show current \n Phase functions', self)
        self.Show_img_PhF_button = QPushButton("Image \n Phase Function", self)

        self.progress_SPF = QProgressBar(self)

        self.Compute_PhF_button.setEnabled(False)
        self.Show_disk_PhF_button.setEnabled(False)
        self.Show_img_PhF_button.setEnabled(False)

        self.Compute_PhF_button.clicked.connect(self.Run_SPF)
        self.Show_disk_PhF_button.clicked.connect(self.Show_disk_PhaseFunction)
        self.Show_img_PhF_button.clicked.connect(self.Show_img_PhF)

        self.Is_computed = QLabel('None')
        self.Is_computed.setStyleSheet('font-size: 14px;')
        self.Is_computed.setFixedHeight(15)

        # Add Button in List
        self.List_Buttons.append(self.browse_button)
        self.List_Buttons.append(self.Display_Fit_button)
        self.List_Buttons.append(self.Compute_Fit_button)
        self.List_Buttons.append(self.R_adjustement)
        self.List_Buttons.append(self.Compute_PhF_button)
        self.List_Buttons.append(self.Show_disk_PhF_button)
        self.List_Buttons.append(self.Show_img_PhF_button)
        
        self.List_Buttons.append(self.auUnitButton)
        self.List_Buttons.append(self.PixelUnitButton)
        self.List_Buttons.append(self.ArcsecUnitButton)
        
        self.List_Buttons.append(self.img_0_button)
        self.List_Buttons.append(self.img_1_button)
        self.List_Buttons.append(self.img_2_button)
        self.List_Buttons.append(self.img_3_button)
        self.List_Buttons.append(self.img_4_button)
        self.List_Buttons.append(self.img_5_button)
        self.List_Buttons.append(self.HeaderButton)

        self.Set_Style_Button()

        # Figure
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_facecolor('black')
        self.ax.set_ylabel('$\Delta$ RA (arcsec)')
        self.ax.set_xlabel('$\Delta$ DEC (arcsec)')
        self.canvas = FigureCanvas(self.fig)
        layout = QHBoxLayout()

        # Browse files
        Browsebox = QFrame()
        Browsebox.setFrameShape(QFrame.StyledPanel)
        Browsebox.setFrameShadow(QFrame.Raised)
        Browsebox.setStyleSheet("background-color: white;")

        browsebox = QVBoxLayout()
        browsebox.addWidget(self.browse_button)
        browsebox.addWidget(self.file_label)
        Browsebox.setLayout(browsebox)

        # Fitting
        FitBox = QFrame()
        FitBox.setFrameShape(QFrame.StyledPanel)
        FitBox.setFrameShadow(QFrame.Raised)
        FitBox.setStyleSheet("background-color: white;")

        FitBoxButton = QVBoxLayout()
        FitBoxButton.addWidget(self.Display_Fit_button)
        FitBoxButton.addWidget(self.Compute_Fit_button)
        FitBoxButton.addWidget(self.fit_file)        

        InclinationBox   = QHBoxLayout()
        PositionAngleBox = QHBoxLayout()
        HeightBox        = QHBoxLayout()
        RadiusBox        = QHBoxLayout()
        AspectRatioBox   = QHBoxLayout()
        AlphaBox         = QHBoxLayout()

        InclinationBox.addWidget(self.Inclination_text)
        InclinationBox.addWidget(self.InclinationLine)
        InclinationBox.addWidget(self.Err_Inclination_text)
        InclinationBox.addWidget(self.ErrInclinationLine)
        PositionAngleBox.addWidget(self.PositionAngle_text)
        PositionAngleBox.addWidget(self.PositionAngleLine)
        PositionAngleBox.addWidget(self.Err_PositionAngle_text)
        PositionAngleBox.addWidget(self.ErrPositionAngleLine)
        HeightBox.addWidget(self.Height_text)
        HeightBox.addWidget(self.HeightLine)
        HeightBox.addWidget(self.Err_Height_text)
        HeightBox.addWidget(self.ErrHeightLine)
        RadiusBox.addWidget(self.Radius_text)
        RadiusBox.addWidget(self.RadiusLine)
        RadiusBox.addWidget(self.Err_Radius_text)
        RadiusBox.addWidget(self.ErrRadiusLine)
        AspectRatioBox.addWidget(self.Scale_Height_text)
        AspectRatioBox.addWidget(self.AspectRatioLine)
        AspectRatioBox.addWidget(self.Err_Scale_Height_text)
        AspectRatioBox.addWidget(self.ErrAspectRatioLine)
        AlphaBox.addWidget(self.Alpha_text)
        AlphaBox.addWidget(self.PowerLawAlphaLine)
        AlphaBox.addWidget(self.Err_Alpha_text)
        AlphaBox.addWidget(self.ErrPowerLawAlphaLine)

        StarPositionBox = QHBoxLayout()
        StarPositionBox.addWidget(self.CheckStar)
        StarPositionBox.addWidget(self.X_StarPositionLabel)
        StarPositionBox.addWidget(self.X_StarPosition)
        StarPositionBox.addWidget(self.Y_StarPositionLabel)
        StarPositionBox.addWidget(self.Y_StarPosition)
        StarPositionBox.addWidget(self.StarPositionLabel)
        
        FitParameters = QVBoxLayout()
        FitParameters.addWidget(self.CheckInputParams)
        FitParameters.addLayout(InclinationBox)
        FitParameters.addLayout(PositionAngleBox)
        FitParameters.addLayout(HeightBox)
        FitParameters.addLayout(RadiusBox)
        FitParameters.addLayout(AspectRatioBox)
        FitParameters.addLayout(AlphaBox)
        # FitParameters.addLayout(StarPositionBox)

        fitbox0 = QHBoxLayout()
        fitbox0.addLayout(FitBoxButton)
        fitbox0.addLayout(FitParameters)
        # FitBox.setLayout(fitbox0)

        fitbox = QVBoxLayout()
        fitbox.addLayout(fitbox0)
        fitbox.addLayout(StarPositionBox)
        FitBox.setLayout(fitbox)

    
        # Parameters
        UnitBox = QHBoxLayout()
        UnitBox.addWidget(self.auUnitButton)
        UnitBox.addWidget(self.ArcsecUnitButton)
        UnitBox.addWidget(self.PixelUnitButton)
        pixelscalebox = QHBoxLayout()
        pixelscalebox.addWidget(self.pixelscale_label)
        pixelscalebox.addWidget(self.pixelscale_entry)
        Distancebox = QHBoxLayout()
        Distancebox.addWidget(self.distance_label)
        Distancebox.addWidget(self.distance_entry)
        R_inbox = QHBoxLayout()
        R_inbox.addWidget(self.R_in_label)
        R_inbox.addWidget(self.R_in_entry)
        R_outbox = QHBoxLayout()
        R_outbox.addWidget(self.R_out_label)
        R_outbox.addWidget(self.R_out_entry)
        nb_binbox = QHBoxLayout()
        nb_binbox.addWidget(self.n_bin_label)
        nb_binbox.addWidget(self.nb_bin_entry)

        LeftParaBox  = QVBoxLayout()

        LeftParaBox.addLayout(pixelscalebox)
        LeftParaBox.addLayout(Distancebox)
        LeftParaBox.addLayout(nb_binbox)

        RightParaBox = QVBoxLayout()
        RightParaBox.addLayout(R_inbox)
        RightParaBox.addLayout(R_outbox)
        RightParaBox.addWidget(self.R_adjustement)

        ParaBox = QFrame()
        ParaBox.setFrameShape(QFrame.StyledPanel)
        ParaBox.setFrameShadow(QFrame.Raised)
        ParaBox.setStyleSheet("background-color: white;")

        parabox = QHBoxLayout()
        parabox.addLayout(LeftParaBox)
        parabox.addLayout(RightParaBox)

        AllParamBox = QVBoxLayout()
        AllParamBox.addLayout(UnitBox)
        AllParamBox.addLayout(parabox)

        ParaBox.setLayout(AllParamBox)

        # Phase Function
        PhFbox = QVBoxLayout()
        PhFBox_button = QHBoxLayout()
        PhFBox_button.addWidget(self.Compute_PhF_button)
        PhFBox_button.addWidget(self.Show_disk_PhF_button)
        PhFBox_button.addWidget(self.Show_img_PhF_button)
        PhFbox.addLayout(PhFBox_button)
        PhFbox.addWidget(self.progress_SPF)
        PhFbox.addWidget(self.Is_computed)

        AllPhFbox = QFrame()
        AllPhFbox.setFrameShape(QFrame.StyledPanel)
        AllPhFbox.setFrameShadow(QFrame.Raised)
        AllPhFbox.setStyleSheet("background-color: white;")

        allphfbox = QVBoxLayout()
        allphfbox.addLayout(PhFbox)
        AllPhFbox.setLayout(allphfbox)

        # View
        h1box = QVBoxLayout()
        h2box = QVBoxLayout()
        h3box = QVBoxLayout()
        h4box = QVBoxLayout()
        h1box.addWidget(self.img_0_button)
        h2box.addWidget(self.img_1_button)
        h3box.addWidget(self.img_2_button)
        h1box.addWidget(self.img_3_button)
        h2box.addWidget(self.img_4_button)
        h3box.addWidget(self.img_5_button)
        h4box.addWidget(self.HeaderButton)

        # h1box.addWidget(self.I_button)
        # h1box.addWidget(self.PI_button)
        # h2box.addWidget(self.stokes_I_button)
        # h2box.addWidget(self.stokes_LPI_button)
        # h3box.addWidget(self.stokes_Qphi_button)
        # h3box.addWidget(self.stokes_Uphi_button)
        # h4box.addWidget(self.stokes_Q_button)
        # h4box.addWidget(self.stokes_U_button)

        # Imshow
        DisplayBox = QFrame()
        DisplayBox.setFixedHeight(150)
        DisplayBox.setFrameShape(QFrame.StyledPanel)
        DisplayBox.setFrameShadow(QFrame.Raised)
        DisplayBox.setStyleSheet("background-color: white;")
        displaybox  = QVBoxLayout()
        displaybox1 = QHBoxLayout()
        displaybox1.addLayout(h1box)
        displaybox1.addLayout(h2box)
        displaybox1.addLayout(h3box)
        displaybox1.addLayout(h4box)
        displaybox.addLayout(displaybox1)
        # displaybox.addWidget(self.Warning_Stokes_1)
        # displaybox.addWidget(self.Warning_Stokes_2)

        DisplayBox.setLayout(displaybox)

        # Organisation
        LeftBBox = QFrame()
        LeftBBox.setFixedWidth(500)
        LeftBBox.setFrameShape(QFrame.StyledPanel)
        LeftBBox.setFrameShadow(QFrame.Raised)
        LeftBBox.setStyleSheet("background-color: lightgrey;")

        leftbbox = QVBoxLayout()
        leftbbox.addWidget(Browsebox)
        leftbbox.addWidget(FitBox)
        leftbbox.addWidget(ParaBox)
        leftbbox.addWidget(AllPhFbox)

        LeftBBox.setLayout(leftbbox)
        
        RightBBox = QFrame()
        RightBBox.setFrameShape(QFrame.StyledPanel)
        RightBBox.setFrameShadow(QFrame.Raised)
        RightBBox.setStyleSheet("background-color: lightgrey;")
        rightbbox = QVBoxLayout()
        rightbbox.addWidget(self.ZoomLabel)
        rightbbox.addWidget(self.ZoomSlider)
        rightbbox.addWidget(self.canvas)
        rightbbox.addWidget(DisplayBox)

        RightBBox.setLayout(rightbbox)

        layout.addWidget(LeftBBox)
        layout.addWidget(RightBBox)
        self.setLayout(layout)

        
        self.setGeometry(100, 100, 1200, 900)
        self.setWindowTitle('GUI Disk Fitting Yields to Scattering Phase Function Extraction')
        
    def Set_Style_Button(self):
        for button in self.List_Buttons:
            if button.isEnabled():
                bg_color = 'lightgrey'
                tx_color = 'Black'
                bd_color = 'grey'
            else :
                bg_color = 'white'
                tx_color = 'grey'
                bd_color = 'lightgrey'
            button.setStyleSheet('QPushButton {'
                                        'color: ' + tx_color + ';'
                                        'background-color: '+ bg_color +';'
                                        'border: 2px solid '+ bd_color +';'
                                        'border-radius: 8px;'
                                        'padding: 8px 16px;'
                                        'font-size: 14px;'
                                        '}'
                                        'QPushButton:hover {'
                                        'background-color: grey;'
                                        '}'
                                        'QPushButton:pressed {'
                                        'border-style: inset;'
                                        '}')
        for frame in self.List_Frame :
            frame.setStyleSheet('QPushButton {'
                                        'color: ;'
                                        'background-color: #4CAF50;'
                                        'border: 2px solid #4CAF50;'
                                        'border-radius: 8px;'
                                        'padding: 8px 16px;'
                                        'font-size: 14px;'
                                        '}'
                                'QPushButton : hover {'
                                        'background-color: #45a049;'
                                        '}'
                                'QPushButton : pressed {'
                                        'border-style: inset;'
                                        '}')
            
    def Set_Initial_Values(self):
        print(self.file_name.replace(' ', '').replace('_', '').replace('-', '').upper())
        self.nb_bin_entry.setEnabled(True)
        self.pixelscale_entry.setEnabled(True)
        self.R_in_entry.setEnabled(True)
        self.R_out_entry.setEnabled(True)
        self.distance_entry.setEnabled(True)
        self.nb_bin_entry.setText('37')  # To have 5° width bin size
        self.lam      = 1.6e-6           # m
        self.Diameter = 8.2              # m
        self.pixelscale_entry.setText('0.01225')
        try :
            Distances = Tools.DiskDistances()
            self.distance_entry.setText(str(Tools.Get_Distance(Distances, self.file_path)))
        except:
            self.distance_entry.setText(str(100))
            
        if "SPHERE" in self.file_name.upper():
            self.Diameter = 8.2         # m
            self.pixelscale_entry.setText('0.01225')
        if "_J" in self.file_name.upper():
            self.lam = 1.2e-6
        elif '_H' in self.file_name.upper():
            self.lam = 1.6e-6

        if "HD 34282" or 'HD34282' in self.file_name:
            self.R_in_entry.setValue(190)
            self.R_out_entry.setValue(220)

        if "LkCa15" in self.file_name:
            self.R_in_entry.setValue(58)
            self.R_out_entry.setValue(94)

        if "V4046" in self.file_name:
            self.R_in_entry.setValue(23)
            self.R_out_entry.setValue(34)

        if "PDS 66" in self.file_name:
            self.R_in_entry.setValue(64)
            self.R_out_entry.setValue(104)

        if "RX J1852" in self.file_name:
            self.R_in_entry.setValue(36)
            self.R_out_entry.setValue(68)

        if "RX J1615" in self.file_name:
            self.R_in_entry.setValue(146)
            self.R_out_entry.setValue(181)

        if "HD163296" in self.file_name:
            self.R_in_entry.setValue(55)
            self.R_out_entry.setValue(75)

        if 'MCFOST_PA' in self.file_name:
            self.lam = 1.6e-6
            self.R_in_entry.setValue(175)
            self.R_out_entry.setValue(185)
            self.pixelscale_entry.setText('0.01225')

        if 'MCFOST_Si_a_' in self.file_name:
            self.lam = 1.6e-6
            self.R_in_entry.setValue(180)
            self.R_out_entry.setValue(280)
            self.pixelscale_entry.setText('0.01996007982')

        if 'MCFOST_AmCa_a_' in self.file_name:
            self.lam = 1.6e-6
            self.R_in_entry.setValue(180)
            self.R_out_entry.setValue(280)
            self.pixelscale_entry.setText('0.01996007982')
        
        if 'VBB' in self.file_name:
            self.pixelscale_entry.setText('0.0036')
        self.pixelscale = float(self.pixelscale_entry.text())
        self.r_beam = self.lam/self.Diameter
        # print(self.Diameter, self.lam)

# ========================================================================================
# ================================    File Finder    =====================================
# ========================================================================================

    def browse_files(self):
        self.Ellipse_Extraction = False
        self.ax.cla()
        self.ax.set_ylabel('$\Delta$RA (arcsec)')
        self.ax.set_xlabel('$\Delta$DEC (arcsec)')
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, 'Select a file', filter='Fichiers FITS (*.fits)')
        self.disk_name = self.file_path.split("/")[-1]
        self.disk_name = self.disk_name[:-5]
        if self.file_path:
            self.Image_Displayed = True
            self.file_name = (self.file_path.split('/'))[-1]
            self.file_label.setText(f"Selected file : {self.file_name}")
            self.Compute_Fit_button.setEnabled(True)
            self.fit_file_name = 'Fitting_' + self.disk_name + '.pkl'
            self.Set_Initial_Values()
            self.Init_image(self.file_path)
            if self.fit_file_name in os.listdir(self.Fitting_Folder):
                # self.Ellipse_Extraction = True
                self.fit_file.setText("Is Fitting already? Yes")
                [self.incl, self.r_ref, self.h_ref, self.aspect, self.alpha, self.PA], [self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, self.D_alpha, self.D_PA] = self.Structure_Params()
                self.InclinationLine.setText(str(np.round(np.degrees(self.incl),2)))
                self.ErrInclinationLine.setText(str(np.round(np.degrees(self.D_incl),2)))
                self.PositionAngleLine.setText(str(np.round(np.degrees(self.PA),2)))
                self.ErrPositionAngleLine.setText(str(np.round(np.degrees(self.D_PA),2)))
                self.HeightLine.setText(str(np.round(self.h_ref,3)))
                self.RadiusLine.setText(str(np.round(self.r_ref,3)))
                self.AspectRatioLine.setText(str(np.round(self.h_ref/self.r_ref**self.alpha,3)))
                self.ErrHeightLine.setText(str(np.round(self.D_h_ref,3)))
                self.ErrRadiusLine.setText(str(np.round(self.D_r_ref,3)))
                if self.h_ref != 0 and self.r_ref != 0:
                    self.AspectRatioLine.setText(str(np.round(self.h_ref/self.r_ref**self.alpha,3)))
                    self.ErrAspectRatioLine.setText(str(np.round((self.h_ref/self.r_ref**self.alpha) * np.sqrt((self.D_r_ref/self.r_ref)**2 + (self.D_h_ref/self.h_ref)**2),3)))
                else :
                    self.r_ref = 1e-20
                    self.h_ref = 1e-20
                    self.RadiusLine.setText(str(self.r_ref))
                    self.HeightLine.setText(str(self.h_ref))
                    self.AspectRatioLine.setText(str(0))
                    self.ErrAspectRatioLine.setText(str(0))
                self.PowerLawAlphaLine.setText("1")
                self.ErrPowerLawAlphaLine.setText("0")
                self.fit_file.setStyleSheet('font-size: 14px; color: green;')
                self.Compute_PhF_button.setEnabled(True)
                self.Display_Fit_button.setEnabled(True)
                self.R_adjustement.setEnabled(True)
            else : 
                # self.Ellipse_Extraction = False
                self.fit_file.setText("Is Fitting already? No")
                self.InclinationLine.setText(" None ")
                self.ErrInclinationLine.setText(" None ")
                self.PositionAngleLine.setText(" None ")
                self.ErrPositionAngleLine.setText(" None ")
                self.HeightLine.setText(" None ")
                self.RadiusLine.setText(" None ")
                self.AspectRatioLine.setText(" None ")
                self.ErrHeightLine.setText(" None ")
                self.ErrRadiusLine.setText(" None ")
                self.ErrAspectRatioLine.setText(" None ")
                self.PowerLawAlphaLine.setText("1")
                self.ErrPowerLawAlphaLine.setText("0")
                self.fit_file.setStyleSheet('font-size: 14px; color: red;')
                self.Compute_PhF_button.setEnabled(False)
                self.Display_Fit_button.setEnabled(False)
                self.R_adjustement.setEnabled(False)
            if self.img_name + '_' + self.disk_name + '.spf' in os.listdir(self.SPF_Folder) :
                self.Is_computed.setText(self.disk_name + " Phase Function is already computed")
                self.Is_computed.setStyleSheet('font-size: 14px; color: green')
                self.Show_disk_PhF_button.setEnabled(True)
                self.Show_img_PhF_button.setEnabled(True)
            else : 
                self.Is_computed.setText(self.disk_name + " Phase Function is not computed")
                self.Is_computed.setStyleSheet('font-size: 14px; color: red')
                self.Show_disk_PhF_button.setEnabled(False)
                self.Show_img_PhF_button.setEnabled(False)
        self.Set_Style_Button()
        if self.Fitting and self.Fitting.isVisible():
            self.Fitting.close()

# ==================================================================
# =====================    Image Display   =========================
# ==================================================================
    def Change_View(self, img_show):
        for idx, img in enumerate(self.all_display):
            if img == img_show:
                self.img_chose    = self.all_img[idx]
                self.thresh_chose = self.all_thresh[idx]
                self.img_name     = self.all_name[idx]
                img.set_visible(True)
                if self.img_name + '_' + self.disk_name + '.spf' in os.listdir(self.SPF_Folder) :
                    self.Is_computed.setText(self.disk_name + " Phase Function is already computed")
                    self.Is_computed.setStyleSheet('font-size: 14px; color: green')
                    self.Show_disk_PhF_button.setEnabled(True)
                    self.Show_img_PhF_button.setEnabled(True)
                else : 
                    self.Is_computed.setText(self.disk_name + " Phase Function is not computed")
                    self.Is_computed.setStyleSheet('font-size: 14px; color: red')
                    self.Show_disk_PhF_button.setEnabled(False)
                    self.Show_img_PhF_button.setEnabled(False)
                self.Set_Style_Button()
            else:
                img.set_visible(False)
        self.Zoom_Slider_Update(self.ZoomSlider.value())

    def Init_image(self, file_path):
        [self.img_0,    self.img_1,    self.img_2,    self.img_3,    self.img_4,    self.img_5], [self.thresh_0, self.thresh_1, self.thresh_2, self.thresh_3, self.thresh_4, self.thresh_5] = Tools.Images_Opener(file_path)
        # self.img_I = Images['I']
        # self.img_Qphi = Images['Q_phi']
        # self.img_Uphi = Images['U_phi']
        # self.img_V = Images['V']
        # self.img_Q = Images['Q']
        # self.img_U = Images['U']
        # self.img_LPI = Images['LP_I']
        # self.Warning_Stokes_2.setText(message)

        PixelScale = float(self.pixelscale_entry.text())
        size = len(self.img_0)/2 * PixelScale
        # self.all_img = [self.stokes_I,  self.stokes_Qphi,  self.stokes_Uphi,  self.stokes_Q,  self.stokes_U,  self.stokes_LPI,  self.all_img]
        self.all_img    = [self.img_0,    self.img_1,    self.img_2,    self.img_3,    self.img_4,    self.img_5]
        self.all_thresh = [self.thresh_0, self.thresh_1, self.thresh_2, self.thresh_3, self.thresh_4, self.thresh_5]
        self.all_name = ["IMG0",  "IMG1", "IMG2",  "IMG3", "IMG4",  "IMG5"]
        # for image in self.all_img:
        #     image = np.abs(image)
        self.img_0_button.setEnabled(True)
        self.img_1_button.setEnabled(True)
        self.img_2_button.setEnabled(True)
        self.img_3_button.setEnabled(True)
        self.img_4_button.setEnabled(True)
        self.img_5_button.setEnabled(True)
        self.HeaderButton.setEnabled(True)
        # self.display_I      = self.ax.imshow(self.stokes_I,    extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=False)
        # self.display_Qphi   = self.ax.imshow(self.stokes_Qphi, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=False)
        # self.display_Uphi   = self.ax.imshow(self.stokes_Uphi, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=False)
        # self.display_Q      = self.ax.imshow(self.stokes_Q,    extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=False)
        # self.display_U      = self.ax.imshow(self.stokes_U,    extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=False)
        # self.display_LPI    = self.ax.imshow(self.stokes_LPI,  extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=False)
        # self.display_img_I  = self.ax.imshow(self.img_I,       extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=False)
        # self.display_img_PI = self.ax.imshow(self.img_PI,      extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", norm=colors.LogNorm(), visible=True)
        
        self.display_0 = self.ax.imshow(self.img_0, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", visible=True,  norm=colors.SymLogNorm(linthresh=self.thresh_0))
        self.display_1 = self.ax.imshow(self.img_1, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", visible=False, norm=colors.SymLogNorm(linthresh=self.thresh_1))
        self.display_2 = self.ax.imshow(self.img_2, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", visible=False, norm=colors.SymLogNorm(linthresh=self.thresh_2))
        self.display_3 = self.ax.imshow(self.img_3, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", visible=False, norm=colors.SymLogNorm(linthresh=self.thresh_3))
        self.display_4 = self.ax.imshow(self.img_4, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", visible=False, norm=colors.SymLogNorm(linthresh=self.thresh_4))
        self.display_5 = self.ax.imshow(self.img_5, extent=[-size, size, -size, size], origin='lower', cmap="gnuplot", visible=False, norm=colors.SymLogNorm(linthresh=self.thresh_5))
        self.img_chose    = self.img_0
        self.thresh_chose = self.thresh_0
        self.img_name  = self.all_name[0]
        # self.all_display = [self.display_I, self.display_Qphi, self.display_Uphi, self.display_Q, self.display_U, self.display_LPI, self.display_img_I, self.display_img_PI]
        self.all_display = [self.display_0, self.display_1, self.display_2, self.display_3, self.display_4, self.display_5]
        self.Zoom_Slider_Update(self.ZoomSlider.value())
        self.CheckStar.setEnabled(True)
        self.X_StarPosition.setText(str(len(self.img_0)/2))
        self.Y_StarPosition.setText(str(len(self.img_0)/2))

    def Zoom_Slider_Update(self, value):
        if self.Image_Displayed:
            PixelScale = float(self.pixelscale_entry.text())
            size  = len(self.img_0) * 100/value
            x_min = len(self.img_0)/2 - size/2
            x_max = len(self.img_0)/2 + size/2
            self.ax.set_xlim((x_min - len(self.img_0)/2)* PixelScale, (x_max - len(self.img_0)/2)* PixelScale)
            self.ax.set_ylim((x_min - len(self.img_0)/2)* PixelScale, (x_max - len(self.img_0)/2)* PixelScale)
            self.canvas.draw()
    
# ==================================================================
# =====================    Fitting Part   ==========================
# ==================================================================
    def on_check_Input_Parameters(self, state):
        if state ==  Qt.Checked:
            self.InputParams = True
            self.Inclination_text.setStyleSheet('font-size: 14px; color: black;')
            self.PositionAngle_text.setStyleSheet('font-size: 14px; color: black;')
            self.Height_text.setStyleSheet('font-size: 14px; color: black;')
            self.Radius_text.setStyleSheet('font-size: 14px; color: black;')
            self.Scale_Height_text.setStyleSheet('font-size: 14px; color: black;')
            self.Alpha_text.setStyleSheet('font-size: 14px; color: black;')
            self.Err_Inclination_text.setStyleSheet('font-size: 14px; color: black;')
            self.Err_PositionAngle_text.setStyleSheet('font-size: 14px; color: black;')
            self.Err_Height_text.setStyleSheet('font-size: 14px; color: black;')
            self.Err_Radius_text.setStyleSheet('font-size: 14px; color: black;')
            self.Err_Scale_Height_text.setStyleSheet('font-size: 14px; color: black;')
            self.Err_Alpha_text.setStyleSheet('font-size: 14px; color: black;')
            self.InclinationLine.setEnabled(True)
            self.PositionAngleLine.setEnabled(True)
            self.HeightLine.setEnabled(True)
            self.RadiusLine.setEnabled(True)
            # self.AspectRatioLine.setEnabled(True)         #  Now, it's R_ref and H_ref that compute tue extraction Zone
            self.PowerLawAlphaLine.setEnabled(True)
            self.ErrInclinationLine.setEnabled(True)
            self.ErrPositionAngleLine.setEnabled(True)
            self.ErrHeightLine.setEnabled(True)
            self.ErrRadiusLine.setEnabled(True)
            # self.ErrAspectRatioLine.setEnabled(True)      #  Now, it's R_ref and H_ref that compute tue extraction Zone
            self.ErrPowerLawAlphaLine.setEnabled(True)
        else:
            self.InputParams = False
            self.Inclination_text.setStyleSheet('font-size: 14px; color: grey;')
            self.PositionAngle_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Height_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Radius_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Scale_Height_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Alpha_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Err_Inclination_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Err_PositionAngle_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Err_Height_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Err_Radius_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Err_Scale_Height_text.setStyleSheet('font-size: 14px; color: grey;')
            self.Err_Alpha_text.setStyleSheet('font-size: 14px; color: grey;')
            self.InclinationLine.setEnabled(False)
            self.PositionAngleLine.setEnabled(False)
            self.HeightLine.setEnabled(False)
            self.RadiusLine.setEnabled(False)
            # self.AspectRatioLine.setEnabled(False)
            self.PowerLawAlphaLine.setEnabled(False)
            self.ErrInclinationLine.setEnabled(False)
            self.ErrPositionAngleLine.setEnabled(False)
            self.ErrHeightLine.setEnabled(False)
            self.ErrRadiusLine.setEnabled(False)
            # self.ErrAspectRatioLine.setEnabled(False)
            self.ErrPowerLawAlphaLine.setEnabled(False)

    def on_check_Star_Position(self, state):
        if state ==  Qt.Checked:
            self.InputStar = True
            self.CheckStar.setStyleSheet('color: black; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
            self.X_StarPositionLabel.setStyleSheet('font-size: 14px; color: black;')
            self.Y_StarPositionLabel.setStyleSheet('font-size: 14px; color: black;')
            self.StarPositionLabel.setStyleSheet('font-size: 14px; color: black;')
            self.X_StarPosition.setEnabled(True)
            self.Y_StarPosition.setEnabled(True)

        else:
            self.InputStar = False
            self.CheckStar.setStyleSheet('color: grey; background-color: white; border: 2px solid lightgrey; border-radius: 8px; font-size: 14px;')
            self.X_StarPositionLabel.setStyleSheet('font-size: 14px; color: grey;')
            self.Y_StarPositionLabel.setStyleSheet('font-size: 14px; color: grey;')
            self.StarPositionLabel.setStyleSheet('font-size: 14px; color: grey;')
            self.X_StarPosition.setText(str(len(self.img_0)/2))
            self.Y_StarPosition.setText(str(len(self.img_0)/2))
            self.X_StarPosition.setEnabled(False)
            self.Y_StarPosition.setEnabled(False)

    def Show_Fitting(self):
        if not self.Fitting or not self.Fitting.isVisible():
            self.Fitting = QWidget()
            self.Fitting.setGeometry(200, 200, 800, 800)
            self.Fitting.setWindowTitle("Fitting")
            figure, ax = plt.subplots(num="Fitting_fig")
            (x_min, x_max) = self.ax.get_xlim()
            PixelScale = float(self.pixelscale_entry.text())
            size = len(self.img_0)/2 * PixelScale
            # x_min, x_max = x_min/PixelScale + len(self.img_PI)/2, x_max/PixelScale + len(self.img_PI)/2
            center = len(self.img_0)/2
            canvas_fit = FigureCanvas(figure)
            layout_fit = QVBoxLayout()
            layout_fit.addWidget(canvas_fit)
            self.Fitting.setLayout(layout_fit)

            [self.incl, self.r_ref, self.h_ref, self.aspect, self.alpha, self.PA], Err = Tools.Load_Structure(self.fit_file_name, Type='Struct')
            [Xc, Yc, X_e, Y_e, a, b, e] = Tools.Load_Structure(self.fit_file_name, Type='Ellipse')
            [X, Y, X_min, X_max, Y_min, Y_max] = Tools.Load_Structure(self.fit_file_name, Type='Points')
            ax.set_title("Numerically Stable Direct \n Least Squares Fitting of Ellipses", fontsize=18, fontweight='bold')
            ax.imshow(self.img_chose, origin='lower', extent=[-size, size, -size, size], cmap="gnuplot", norm=colors.SymLogNorm(linthresh=self.thresh_chose))
            ax.plot((Y_min-center)*PixelScale,    (X_min-center)*PixelScale,   '.', label="range position", color='lightblue', alpha=0.4) # given points
            ax.plot((Y_max-center)*PixelScale,    (X_max-center)*PixelScale,   '.',                         color='lightblue', alpha=0.4) # given points
            ax.plot((Y-center)*PixelScale,    (X-center)*PixelScale,           '.',                         color='blue',      alpha=0.4) # given points
            ax.plot((Y_e-center)*PixelScale,  (X_e-center)*PixelScale,       label="ellipse fit", color='blue')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ax.set_xlabel('X position [pix]')
            ax.set_ylabel('Y position [pix]')
            ax.legend(loc='upper right')
            # ax.text(1, 1, u'i = {:<15} \n PA = {:<15} \n h/r = {:<15}'.format(str(np.round(np.degrees(self.incl),3))+' °', str(np.round(np.degrees(self.PA),3))+' °', str(np.round(self.aspect, 3))), fontsize=15, color='red', ha='left', va='bottom')
            self.Fitting.show()

    def update_params(self):
        try:
            r_ref   = float(self.RadiusLine.text())
            h_ref   = float(self.HeightLine.text())
            D_r_ref = float(self.ErrRadiusLine.text())
            D_h_ref = float(self.ErrHeightLine.text())
            alpha   = float(self.PowerLawAlphaLine.text()) 
            if r_ref != 0 and h_ref != 0 :
                self.AspectRatioLine.setText(str(np.round(h_ref/r_ref**alpha,4)))
                self.ErrAspectRatioLine.setText(str(np.round((h_ref/r_ref**alpha) * np.sqrt((D_r_ref/r_ref)**2 + (D_h_ref/h_ref)**2),4)))
            else:
                r_ref = 1e-20
                h_ref = 1e-20
                self.AspectRatioLine.setText(str(0))
                self.ErrAspectRatioLine.setText(str(0))                
        except ValueError:
            # Si l'entrée n'est pas un nombre, ne rien faire ou afficher un message
            self.AspectRatioLine.setText('None')
            self.ErrAspectRatioLine.setText('None')
# ==================================================================
# ===================    Data Filtering    =========================
# ==================================================================


    def Launch_Filtering_Data(self):
        (x_min, x_max) = self.ax.get_xlim() 
        PixelScale = float(self.pixelscale_entry.text())
        x_min, x_max = x_min/PixelScale + len(self.img_0)/2, x_max/PixelScale + len(self.img_0)/2
        self.Filtering_Window = FilteringWindow(self.disk_name, self.img_chose, self.thresh_chose, x_min, x_max)
        # self.Filtering_Window = FilteringWindow(self.disk_name, self.stokes_Qphi, x_min, x_max)
        self.Filtering_Window.exec_()
        if self.fit_file_name in os.listdir(self.Fitting_Folder):
            self.Display_Fit_button.setEnabled(True)
            self.fit_file.setText("Is Fitting already? Yes")
            self.fit_file.setStyleSheet('font-size: 14px; color: green;')
            [self.incl, self.r_ref, self.h_ref, self.aspect, self.alpha, self.PA], [self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, self.D_alpha, self.D_PA] = self.Structure_Params()
            self.InclinationLine.setText(str(np.round(np.degrees(self.incl),2)))
            self.ErrInclinationLine.setText(str(np.round(np.degrees(self.D_incl),2)))
            self.PositionAngleLine.setText(str(np.round(np.degrees(self.PA),2)))
            self.ErrPositionAngleLine.setText(str(np.round(np.degrees(self.D_PA),2)))
            self.HeightLine.setText(str(np.round(self.h_ref,3)))
            self.RadiusLine.setText(str(np.round(self.r_ref,3)))
            # self.AspectRatioLine.setText(str(np.round(self.h_ref/self.r_ref**self.alpha,4)))
            self.ErrHeightLine.setText(str(np.round(self.D_h_ref,3)))
            self.ErrRadiusLine.setText(str(np.round(self.D_r_ref,3)))
            if self.h_ref != 0 and self.r_ref != 0:
                self.AspectRatioLine.setText(str(np.round(self.h_ref/self.r_ref**self.alpha,3)))
                self.ErrAspectRatioLine.setText(str(np.round((self.h_ref/self.r_ref**self.alpha) * np.sqrt((self.D_r_ref/self.r_ref)**2 + (self.D_h_ref/self.h_ref)**2),3)))
            else :
                self.r_ref = 1e-20
                self.h_ref = 1e-20
                
                self.RadiusLine.setText(str(self.r_ref))
                self.HeightLine.setText(str(self.h_ref))
                self.AspectRatioLine.setText(str(0))
                self.ErrAspectRatioLine.setText(str(0))
            self.PowerLawAlphaLine.setText("1")
            self.ErrPowerLawAlphaLine.setText("0")
            self.Compute_PhF_button.setEnabled(True)

            self.R_adjustement.setEnabled(True)
            self.Set_Style_Button()

    def Structure_Params(self):
        Fit_Name = 'Fitting_' + self.disk_name + '.pkl'
        if self.InputParams:
            self.incl     = np.radians(float(self.InclinationLine.text()))
            self.D_incl   = np.radians(float(self.ErrInclinationLine.text()))
            self.PA       = np.radians(float(self.PositionAngleLine.text()))
            self.D_PA     = np.radians(float(self.ErrPositionAngleLine.text()))
            self.h_ref        = float(self.HeightLine.text())
            self.r_ref   = float(self.RadiusLine.text())
            self.aspect   = float(self.AspectRatioLine.text())
            self.D_h_ref      = float(self.ErrHeightLine.text())
            self.D_r_ref = float(self.ErrRadiusLine.text())
            self.D_aspect = float(self.ErrAspectRatioLine.text())
            self.alpha    = float(self.PowerLawAlphaLine.text())
            self.D_alpha  = float(self.ErrPowerLawAlphaLine.text())
            values = [  self.incl, self.r_ref,    self.h_ref,   self.aspect,   self.alpha,   self.PA]
            errors = [self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, self.D_alpha, self.D_PA]
        else :
            values, errors = Tools.Load_Structure(Fit_Name, Type='Struct')
        return values, errors
# ==================================================================
# ===================    Extraction Zone   =========================
# ==================================================================
    def Ring_Adjust_2(self, state):
        if state == Qt.Checked:
            self.Ellipse_Extraction = True
            self.Extraction_Zone()
        else:
            self.Ellipse_Extraction = False
            self.ellipse_in.remove()
            self.ellipse_out.remove()
            self.ellipse_in = False
            self.canvas.draw()
            
    def Extraction_Zone(self):
        try :
            r_ref   = float(self.RadiusLine.text())
            h_ref   = float(self.HeightLine.text())
            D_r_ref = float(self.ErrRadiusLine.text())
            D_h_ref = float(self.ErrHeightLine.text())
            alpha   = float(self.PowerLawAlphaLine.text())
            incl    = float(self.InclinationLine.text())
            PA      = float(self.PositionAngleLine.text())
            GoodParams = True
        except ValueError:
            GoodParams = False    

        if self.Ellipse_Extraction and GoodParams :
                if self.ellipse_in != False:
                    self.ellipse_in.remove()
                    self.ellipse_out.remove()
                R_in          = int(self.R_in_entry.value())
                R_out         = int(self.R_out_entry.value())
                pixelscale    = float(self.pixelscale_entry.text())
                size          = len(self.img_0)
                # arcsec_extent = size * pixelscale

                X_in, Y_in    = self.Ellipse(R_in, pixelscale) #*1.996007984)
                X_out, Y_out  = self.Ellipse(R_out, pixelscale) #*1.996007984)
                self.ellipse_in  = self.ax.scatter(X_in, Y_in, c='gold', s=1)
                self.ellipse_out = self.ax.scatter(X_out, Y_out, c='gold', s=1)
                self.canvas.draw()

    def Ellipse(self, R, pixelscale):
        [self.incl, self.r_ref, self.h_ref, self.aspect, self.alpha, self.PA], [self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, self.D_alpha, self.D_PA] = self.Structure_Params()
        [Xc, Yc, _, _, _, _, _] = Tools.Load_Structure('Fitting_' + self.disk_name + '.pkl', Type='Ellipse')
        xs = ys = len(self.img_0)/2
        yc_p, xc_p = Tools.Orthogonal_Prejection((xs, ys), (Yc, Xc), np.pi/2 - self.PA)
        Phi = np.linspace(0, 359, 360)
        d = float(self.distance_entry.text())
        FoV_au = d * (648000/np.pi) * np.tan(pixelscale*len(self.img_0)/3600 * np.pi/180)
        pixelscale_au = FoV_au/len(self.img_0)
        R_ref = self.r_ref/pixelscale_au
        H_ref = self.h_ref/pixelscale_au
        R = R / pixelscale_au
        x     = R * np.sin(Phi)
        y     = H_ref * (R/R_ref)**self.alpha * np.sin(self.incl) - R * np.cos(Phi) * np.cos(self.incl)
        x_rot = (x * np.cos(np.pi - self.PA) - y * np.sin(np.pi - self.PA) + (xc_p-xs)) *pixelscale
        y_rot = (x * np.sin(np.pi - self.PA) + y * np.cos(np.pi - self.PA) + (yc_p-ys)) *pixelscale
        return y_rot, x_rot
    
    def Compute_Side(self):
        Side = np.zeros_like(self.img_0)
        Side_imshow = np.zeros((len(self.img_0), len(self.img_0), 3), dtype=np.uint8)
        x0 = y0 = len(self.img_0)/2
        slope = np.tan(self.PA)
        intercept = y0 - x0*slope
        mask_idx = np.indices((len(self.img_0), len(self.img_0)))
        mask = mask_idx[0] > slope * mask_idx[1] + intercept
        inv_mask = mask_idx[0] < slope * mask_idx[1] + intercept
        Side[mask] = 1
        Side_imshow[mask]     = [255, 0, 0]     #red
        Side_imshow[inv_mask] = [0, 0, 255]     #blue
        return Side_imshow

# ==================================================================
# ===================    Phase Functions   =========================
# ==================================================================

    def Run_SPF(self):
        distance = float(self.distance_entry.text())
        R_in     = float(self.R_in_entry.value())
        R_out    = float(self.R_out_entry.value())
        n_bin    = int(self.nb_bin_entry.text())
        pixelscale = float(self.pixelscale_entry.text())
        values, errors = self.Structure_Params()
        xs = float(self.X_StarPosition.text())
        ys = float(self.Y_StarPosition.text())
        if self.InputStar:
            [Xc, Yc, _, _, _, _, _] = Tools.Load_Structure('Fitting_' + self.disk_name + '.pkl', Type='Ellipse')
            print(np.sqrt((xs-Xc)**2 + (ys-Yc)**2))
        else:
            Xc, Yc = None, None
    
        incl       = values[0]
        R_ref      = values[1]
        H_ref      = values[2]
        aspect     = values[3]
        alpha      = values[4]
        PA         = values[5]
        Err_incl   = errors[0]
        Err_R_ref  = errors[1]
        Err_H_ref  = errors[2]
        Err_aspect = errors[3]
        Err_alpha  = errors[4]
        Err_PA     = errors[5]

        # [Xc, Yc, _, _, _, _, _] = Tools.Load_Structure('Fitting_' + self.disk_name + '.pkl', Type='Ellipse')

        Min_PA = PA - Err_PA
        Max_PA = PA + Err_PA

        Min_incl = incl - Err_incl
        Max_incl = incl + Err_incl

        Min_aspect = aspect - Err_aspect
        Max_aspect = aspect + Err_aspect
        # POOL
        start = time.time()
        self.total_steps = (2 * 300*360 + 3*len(self.img_0)*len(self.img_0) + 2*n_bin) * 7
        self.computation_step = 0
        list_params_SPF = [ [self.img_chose, distance, pixelscale, self.r_beam, (xs, ys), (Xc, Yc), incl,     PA,     R_ref, H_ref, aspect,        alpha, Err_incl, Err_R_ref, Err_H_ref, Err_aspect, R_in, R_out, n_bin, "Total"],
                            [self.img_chose, distance, pixelscale, self.r_beam, (xs, ys), (Xc, Yc), Min_incl, PA,     R_ref, H_ref, aspect,        alpha, Err_incl, Err_R_ref, Err_H_ref, Err_aspect, R_in, R_out, n_bin, "Incl"],
                            [self.img_chose, distance, pixelscale, self.r_beam, (xs, ys), (Xc, Yc), Max_incl, PA,     R_ref, H_ref, aspect,        alpha, Err_incl, Err_R_ref, Err_H_ref, Err_aspect, R_in, R_out, n_bin, "Incl"],
                            [self.img_chose, distance, pixelscale, self.r_beam, (xs, ys), (Xc, Yc), incl,     Min_PA, R_ref, H_ref, aspect,        alpha, Err_incl, Err_R_ref, Err_H_ref, Err_aspect, R_in, R_out, n_bin, "PA"],
                            [self.img_chose, distance, pixelscale, self.r_beam, (xs, ys), (Xc, Yc), incl,     Max_PA, R_ref, H_ref, aspect,        alpha, Err_incl, Err_R_ref, Err_H_ref, Err_aspect, R_in, R_out, n_bin, "PA"],
                            [self.img_chose, distance, pixelscale, self.r_beam, (xs, ys), (Xc, Yc), incl,     PA,     R_ref, H_ref, Min_aspect,    alpha, Err_incl, Err_R_ref, Err_H_ref, Err_aspect, R_in, R_out, n_bin, "Aspect"],
                            [self.img_chose, distance, pixelscale, self.r_beam, (xs, ys), (Xc, Yc), incl,     PA,     R_ref, H_ref, Max_aspect,    alpha, Err_incl, Err_R_ref, Err_H_ref, Err_aspect, R_in, R_out, n_bin, "Aspect"]]
        SPF.Compute_SPF(list_params_SPF, self.file_name, self.img_name)
        end = time.time()
        self.Is_computed.setText(" SPF computed - time = " +str(np.round(end-start, 2)) + ' seconds')
        self.Is_computed.setStyleSheet('font-size: 14px; color: green')
        self.Show_disk_PhF_button.setEnabled(True)
        self.Show_img_PhF_button.setEnabled(True)
        self.Set_Style_Button()

# ==================================================================
# ===================    Display Results   =========================
# ==================================================================

    def Show_disk_PhaseFunction(self):
        self.ShowNormalized = False
        self.LBremoved = True
        self.ShowSide = True
        self.LogScale = False
        self.ShowMCFOST = False
        self.Disk_PhF = QWidget()
        self.Disk_PhF.setWindowTitle("Phase Functions")

        figure_Disk_PhF, (self.ax_Disk_PhF_I, self.ax_Disk_PhF_PI, self.ax_Disk_DoP) = plt.subplots(1, 3)
        self.ax_Disk_DoP.clear()
        self.ax_Disk_PhF_I.clear()
        self.ax_Disk_PhF_PI.clear()
        self.ax_Disk_PhF_I.set_xlabel("Total Flux")
        self.ax_Disk_PhF_I.set_ylabel("Scattering Angle (deg)")
        self.ax_Disk_PhF_PI.set_xlabel("Polarized Flux")
        self.ax_Disk_PhF_PI.set_ylabel("Scattering Angle (deg)")
        self.ax_Disk_DoP.set_xlabel("Degree of Polarization")
        self.ax_Disk_DoP.set_ylabel("Scattering Angle (deg)")
        self.canvas_Disk_PhF = FigureCanvas(figure_Disk_PhF)
        Toolbar = NavigationToolbar(self.canvas_Disk_PhF, self)
        layout_Disk_PhF = QVBoxLayout()


        self.NormButton   = QCheckBox("Normalize")
        LogButton    = QCheckBox("LogScale")
        LB_Remove    = QCheckBox("Limb Brightening Corrected")
        SideButton   = QCheckBox("Each Side")
        MCFOSTButton = QCheckBox("MCFOST")

        LB_Remove.setChecked(True)
        SideButton.setChecked(True)

        if 'MCFOST' not in self.file_name:
            MCFOSTButton.setEnabled(False)
        MCFOSTButton.stateChanged.connect(self.MCFOSTCheck)
        self.NormButton.stateChanged.connect(self.NormalizeCheck)

        LogButton.stateChanged.connect(self.LogCheck)
        LB_Remove.stateChanged.connect(self.LBCheck)
        SideButton.stateChanged.connect(self.SideCheck)

        # folder = os.getcwd() + '\Results\Phase_Function\MaxSoft/' + self.img_name + '_' + self.disk_name + '.spf'
        folder = f"{self.SPF_Folder}/{self.img_name}_{self.disk_name}.spf"
        self.Scatt,      self.I,      self.PI,      self.Err_Scatt,      self.Err_I,      self.Err_PI,      self.LB     , self.Err_LB      = Tools.Get_PhF(folder, side='All')
        self.Scatt_east, self.I_east, self.PI_east, self.Err_Scatt_east, self.Err_I_east, self.Err_PI_east, self.LB_east, self.Err_LB_east = Tools.Get_PhF(folder, side='East')
        self.Scatt_west, self.I_west, self.PI_west, self.Err_Scatt_west, self.Err_I_west, self.Err_PI_west, self.LB_west, self.Err_LB_west = Tools.Get_PhF(folder, side='West')
        # print(self.PI)
        self.I_Displayed    = self.ax_Disk_PhF_I.errorbar(self.Scatt, self.I, xerr=self.Err_Scatt, yerr=np.abs(self.Err_I), color='black', label='all disk')
        self.PI_Displayed   = self.ax_Disk_PhF_PI.errorbar(self.Scatt, self.PI, xerr=self.Err_Scatt, yerr=np.abs(self.Err_PI), color='black', label='all disk')
        self.DoP_Displayed  = self.ax_Disk_DoP.errorbar(self.Scatt, self.PI/self.I, xerr=self.Err_Scatt, yerr=np.sqrt((self.Err_PI/self.PI)**2 + (self.Err_I/self.I)**2), color='black', label='all disk')
        
        self.I_east_Displayed    = self.ax_Disk_PhF_I.errorbar(self.Scatt_east,              self.I_east,       xerr=self.Err_Scatt_east,       yerr=np.abs(self.Err_I_east),                                                                   color='blue', label='east side', alpha=0.4, ls='dotted')
        self.PI_east_Displayed   = self.ax_Disk_PhF_PI.errorbar(self.Scatt_east,             self.PI_east,       xerr=self.Err_Scatt_east,       yerr=np.abs(self.Err_PI_east),                                                                  color='blue', label='east side', alpha=0.4, ls='dotted')
        self.DoP_east_Displayed  = self.ax_Disk_DoP.errorbar(  self.Scatt_east, self.PI_east/self.I_east,       xerr=self.Err_Scatt_east,       yerr=np.sqrt((self.Err_PI_east/self.PI_east)**2 + (self.Err_I_east/self.I_east)**2),    color='blue', label='east side', alpha=0.4, ls='dotted')
        
        self.I_west_Displayed    = self.ax_Disk_PhF_I.errorbar(self.Scatt_west,              self.I_west,       xerr=self.Err_Scatt_west,       yerr=np.abs(self.Err_I_west),                                                                   color='red', label='west side', alpha=0.4, ls='dotted')
        self.PI_west_Displayed   = self.ax_Disk_PhF_PI.errorbar(self.Scatt_west,             self.PI_west,       xerr=self.Err_Scatt_west,       yerr=np.abs(self.Err_PI_west),                                                                  color='red', label='west side', alpha=0.4, ls='dotted')
        self.DoP_west_Displayed  = self.ax_Disk_DoP.errorbar(  self.Scatt_west, self.PI_west/self.I_west,       xerr=self.Err_Scatt_west,       yerr=np.sqrt((self.Err_PI_west/self.PI_west)**2 + (self.Err_I_west/self.I_west)**2),    color='red', label='west side', alpha=0.4, ls='dotted')
        self.Side_Displayed = True

        self.MCFOST_Displayed = False


        I_bound   = np.concatenate([line.get_children()[0].get_ydata() for line in [self.I_Displayed, self.I_east_Displayed, self.I_west_Displayed]])
        PI_bound  = np.concatenate([line.get_children()[0].get_ydata() for line in [self.PI_Displayed, self.PI_east_Displayed, self.PI_west_Displayed]])
        DoP_bound = np.concatenate([line.get_children()[0].get_ydata() for line in [self.DoP_Displayed, self.DoP_east_Displayed, self.DoP_west_Displayed]])
        marge = 0.1
        self.ax_Disk_PhF_I.set_ylim(np.min(I_bound)*(1-marge), np.max(I_bound)*(1+marge))
        self.ax_Disk_PhF_PI.set_ylim(np.min(PI_bound)*(1-marge), np.max(PI_bound)*(1+marge))
        self.ax_Disk_DoP.set_ylim(np.min(DoP_bound)*(1-marge), np.max(DoP_bound)*(1+marge))

        layout_Disk_PhF.addWidget(Toolbar)
        layout_Disk_PhF.addWidget(self.canvas_Disk_PhF)

        checkBox = QHBoxLayout()
        checkBox.addWidget(self.NormButton)
        checkBox.addWidget(LB_Remove)
        checkBox.addWidget(SideButton)
        checkBox.addWidget(LogButton)
        checkBox.addWidget(MCFOSTButton)
        layout_Disk_PhF.addLayout(checkBox)
        self.Disk_PhF.setLayout(layout_Disk_PhF)
        self.Disk_PhF.show()

    def NormalizeCheck(self, state):
        if state == Qt.Checked:
            self.ShowNormalized = True
            self.ax_Disk_PhF_I.autoscale()
        else:
            self.ShowNormalized = False
            self.ax_Disk_PhF_I.autoscale()
        self.Update_Show_PhF()
    
    def LBCheck(self, state):
        if state == Qt.Checked:
            self.LBremoved = True
        else:
            self.LBremoved = False
        self.Update_Show_PhF()
        
    def SideCheck(self, state):
        if state == Qt.Checked:
            self.ShowSide = True
        else:
            self.ShowSide = False
        self.Update_Show_PhF()
    
    def LogCheck(self, state):
        if state == Qt.Checked:
            self.LogScale = True
            self.ax_Disk_PhF_I.set_yscale('log')
            self.ax_Disk_PhF_PI.set_yscale('log')
        else:
            self.ShowSide = False
            self.ax_Disk_PhF_I.set_yscale('linear')
            self.ax_Disk_PhF_PI.set_yscale('linear')
        self.Update_Show_PhF()

    def MCFOSTCheck(self, state):
        if state == Qt.Checked:
            self.ShowMCFOST = True
            self.ShowNormalized = True
            self.NormButton.setChecked(True)
            self.NormButton.setEnabled(False)
            self.ax_Disk_PhF_I.autoscale()
        else:
            self.ShowMCFOST = False
            self.NormButton.setEnabled(True)
        self.Update_Show_PhF()

    def Update_Show_PhF(self):
        Coeff_all_I,  Coeff_all_PI  = 1, 1
        Coeff_east_I, Coeff_east_PI = 1, 1
        Coeff_west_I, Coeff_west_PI = 1, 1
        self.I_Displayed.remove()
        self.PI_Displayed.remove()
        self.DoP_Displayed.remove()

        if self.ShowMCFOST:
            MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, MCFOST_Err_I, MCFOST_Err_PI, MCFOST_Err_DoP = Tools.MCFOST_PhaseFunction('/'.join(self.file_path.split('/')[:-2]), self.file_name[:-5], self.normalization)
            MCFOST_I  = np.abs(Tools.Same90(90, self.Scatt,  self.I/(self.LB * np.max(self.I)),  MCFOST_Scatt, MCFOST_I))
            MCFOST_PI = np.abs(Tools.Same90(90, self.Scatt, self.PI/(self.LB * np.max(self.PI)), MCFOST_Scatt, MCFOST_PI))
            self.I_MCFOST_Displayed   = self.ax_Disk_PhF_I.errorbar( MCFOST_Scatt, MCFOST_I,           color='purple', label='MCFOST', alpha=0.4, ls='dashed')
            self.PI_MCFOST_Displayed  = self.ax_Disk_PhF_PI.errorbar(MCFOST_Scatt, MCFOST_PI,          color='purple', label='MCFOST', alpha=0.4, ls='dashed')
            self.DoP_MCFOST_Displayed = self.ax_Disk_DoP.errorbar(   MCFOST_Scatt, np.abs(MCFOST_DoP), color='purple', label='MCFOST', alpha=0.4, ls='dashed')
            self.MCFOST_Displayed = True
        else :
            if self.MCFOST_Displayed:
                self.I_MCFOST_Displayed.remove()
                self.PI_MCFOST_Displayed.remove()
                self.DoP_MCFOST_Displayed.remove()
            self.MCFOST_Displayed = False

        if self.Side_Displayed:
            self.I_east_Displayed.remove()
            self.I_west_Displayed.remove()
            self.PI_east_Displayed.remove()
            self.PI_west_Displayed.remove()
            self.DoP_east_Displayed.remove()
            self.DoP_west_Displayed.remove()

        if self.LBremoved:
            Coeff_all_I,  Coeff_all_PI  = Coeff_all_I  * self.LB,        Coeff_all_PI   * self.LB
            Coeff_east_I, Coeff_east_PI = Coeff_east_I * self.LB_east,   Coeff_east_PI  * self.LB_east
            Coeff_west_I, Coeff_west_PI = Coeff_west_I * self.LB_west,   Coeff_west_PI  * self.LB_west
        
        if self.ShowNormalized:
            Coeff_all_I,  Coeff_all_PI  = Coeff_all_I  * np.max(self.I/Coeff_all_I),         Coeff_all_PI   * np.max(self.PI/Coeff_all_PI)
            Coeff_east_I, Coeff_east_PI = Coeff_east_I * np.max(self.I_east/Coeff_east_I),   Coeff_east_PI  * np.max(self.PI_east/Coeff_east_PI)
            Coeff_west_I, Coeff_west_PI = Coeff_west_I * np.max(self.I_west/Coeff_west_I),   Coeff_west_PI  * np.max(self.PI_west/Coeff_west_PI)

        I,      I_east,      I_west      = self.I.copy()/Coeff_all_I,        self.I_east.copy()/Coeff_east_I,        self.I_west.copy()/Coeff_west_I
        PI,     PI_east,     PI_west     = self.PI.copy()/Coeff_all_PI,      self.PI_east.copy()/Coeff_east_PI,      self.PI_west.copy()/Coeff_west_PI
        Err_I,  Err_I_east,  Err_I_west  = self.Err_I.copy()/Coeff_all_I,    self.Err_I_east.copy()/Coeff_east_I,    self.Err_I_west.copy()/Coeff_west_I
        Err_PI, Err_PI_east, Err_PI_west = self.Err_PI.copy()/Coeff_all_PI,  self.Err_PI_east.copy()/Coeff_east_PI,  self.Err_PI_west.copy()/Coeff_west_PI

        self.I_Displayed    = self.ax_Disk_PhF_I.errorbar(self.Scatt,   I, xerr=self.Err_Scatt, yerr=np.abs(Err_I),  color='black', label='all disk')
        self.PI_Displayed   = self.ax_Disk_PhF_PI.errorbar(self.Scatt, PI, xerr=self.Err_Scatt, yerr=np.abs(Err_PI), color='black', label='all disk')
        self.DoP_Displayed  = self.ax_Disk_DoP.errorbar(self.Scatt,  PI/I, xerr=self.Err_Scatt, yerr=np.sqrt((Err_PI/PI)**2 + (Err_I/I)**2), color='black', label='all disk')
                
        if self.ShowSide:
            self.I_east_Displayed   = self.ax_Disk_PhF_I.errorbar(self.Scatt_east,           I_east,  xerr=self.Err_Scatt_east,   yerr=np.abs(Err_I_east),                                                  color='blue', label='east side', alpha=0.4, ls='dotted')
            self.PI_east_Displayed  = self.ax_Disk_PhF_PI.errorbar(self.Scatt_east,         PI_east,  xerr=self.Err_Scatt_east,   yerr=np.abs(Err_PI_east),                                                 color='blue', label='east side', alpha=0.4, ls='dotted')
            self.DoP_east_Displayed = self.ax_Disk_DoP.errorbar(self.Scatt_east,     PI_east/I_east,  xerr=self.Err_Scatt_east,   yerr=np.sqrt((Err_PI_east/PI_east)**2 + (Err_I_east/I_east)**2),  color='blue', label='east side', alpha=0.4, ls='dotted')
            
            self.I_west_Displayed   = self.ax_Disk_PhF_I.errorbar(self.Scatt_west,           I_west,  xerr=self.Err_Scatt_west,   yerr=np.abs(Err_I_west),                                                  color='red', label='west side', alpha=0.4, ls='dotted')
            self.PI_west_Displayed  = self.ax_Disk_PhF_PI.errorbar(self.Scatt_west,         PI_west,  xerr=self.Err_Scatt_west,   yerr=np.abs(Err_PI_west),                                                 color='red', label='west side', alpha=0.4, ls='dotted')
            self.DoP_west_Displayed = self.ax_Disk_DoP.errorbar(self.Scatt_west,     PI_west/I_west,  xerr=self.Err_Scatt_west,   yerr=np.sqrt((Err_PI_west/PI_west)**2 + (Err_I_west/I_west)**2),  color='red', label='west side', alpha=0.4, ls='dotted')
            self.Side_Displayed = True
        else : 
            self.Side_Displayed = False
        if self.Side_Displayed:
            I_bound   = np.concatenate([line.get_children()[0].get_ydata() for line in [self.I_Displayed, self.I_east_Displayed, self.I_west_Displayed]])
            PI_bound  = np.concatenate([line.get_children()[0].get_ydata() for line in [self.PI_Displayed, self.PI_east_Displayed, self.PI_west_Displayed]])
            DoP_bound = np.concatenate([line.get_children()[0].get_ydata() for line in [self.DoP_Displayed, self.DoP_east_Displayed, self.DoP_west_Displayed]])
        else : 
            I_bound   = self.I_Displayed.get_children()[0].get_ydata()
            PI_bound  = self.PI_Displayed.get_children()[0].get_ydata()
            DoP_bound = self.DoP_Displayed.get_children()[0].get_ydata()

        marge = 0.1
        self.ax_Disk_PhF_I.set_ylim(np.min(I_bound)*(1-marge), np.max(I_bound)*(1+marge))
        self.ax_Disk_PhF_PI.set_ylim(np.min(PI_bound)*(1-marge), np.max(PI_bound)*(1+marge))
        self.ax_Disk_DoP.set_ylim(np.min(DoP_bound)*(1-marge), np.max(DoP_bound)*(1+marge))
        self.canvas_Disk_PhF.draw()
    
    def Ginski_Color(self, Disks):
        if 'LkCa15' in Disks:
            c = 'blue'
        elif 'HD163296' in Disks:
            c = 'red'
        elif 'RX J1615' in Disks:
            c = 'coral'
        elif 'PDS 66' in Disks:
            c = 'khaki'
        elif 'HD34282' in Disks:

            c = 'black'
        elif 'RX J1852' in Disks:
            c = 'orange'
        elif 'V4046' in Disks:
            c = 'green'
        return c

    def Show_img_PhF(self):
        self.Img_PhF = QWidget()
        self.Img_PhF.setWindowTitle("Phase Functions")

        Fig_img_phF = plt.figure(figsize = (9, 4))#, dpi = 300)
        # ax_img              = plt.subplot2grid((2, 2), (0, 0))
        ax_img_Extraction   = plt.subplot2grid((1, 5), (0, 0), colspan=2, rowspan=1)
        ax_phF              = plt.subplot2grid((1, 5), (0, 2), colspan=3, rowspan=1)

        # Fig_img_phF, (ax_img_Extraction, ax_phF) = plt.subplots(1, 2, figsize=(10,5))
        # Fig_img_phF, (ax_img, ax_img_Extraction, ax_phF) = plt.subplots(1, 3, figsize=(16, 4.5))
        # Fig_img_phF = plt.figure()

        ax_phF.clear()
        # ax_img.clear()
        ax_img_Extraction.clear()

        
        
        Canvas_Img_PhF = FigureCanvas(Fig_img_phF)
        Toolbar = NavigationToolbar(Canvas_Img_PhF, self)
        layout_Img_PhF = QVBoxLayout()
        # folder = os.getcwd() + '\Results\Phase_Function\MaxSoft/' + self.img_name + '_' + self.disk_name + ".spf"
        folder = f"{self.SPF_Folder}/{self.img_name}_{self.disk_name}.spf"
        Scatt,      _, PI,      Err_Scatt, _,      Err_PI,      LB,      Err_LB      = Tools.Get_PhF(folder, side='All')
        Scatt_east, _, PI_east, Err_Scatt_east, _, Err_PI_east, LB_east, Err_LB_east = Tools.Get_PhF(folder, side='East')
        Scatt_west, _, PI_west, Err_Scatt_west, _, Err_PI_west, LB_west, Err_LB_west = Tools.Get_PhF(folder, side='West')

        PI_LB        = PI/LB
        Err_PI_LB    = PI_LB  * np.sqrt((Err_PI/PI)**2   + (Err_LB/LB)**2)
        PI_east      = PI_east/LB_east
        Err_PI_east  = PI_east  * np.sqrt((Err_PI_east/PI_east)**2   + (Err_LB_east/LB_east)**2)
        PI_west      = PI_west/LB_west
        Err_PI_west  = PI_west  * np.sqrt((Err_PI_west/PI_west)**2   + (Err_LB_west/LB_west)**2)
 
        normPI      = np.interp(90, Scatt, PI)
        normPI_LB   = np.interp(90, Scatt, PI_LB)
        normPI_east = np.interp(90, Scatt_east, PI_east)
        normPI_west = np.interp(90, Scatt_west, PI_west)


        PI,      Err_PI      = PI/(normPI),           Err_PI/(normPI)
        PI_LB,   Err_PI_LB   = PI_LB/(normPI_LB),     Err_PI_LB/(normPI_LB)
        PI_east, Err_PI_east = PI_east/(normPI_east), Err_PI_east/(normPI_east)
        PI_west, Err_PI_west = PI_west/(normPI_west), Err_PI_west/(normPI_west)


        R_in = int(self.R_in_entry.value())
        R_out = int(self.R_out_entry.value())
        Side = self.Compute_Side()
        pixelscale = float(self.pixelscale_entry.text())
        size = len(self.img_0)
        arcsec_extent = size/2 * pixelscale
        ax_img_Extraction.imshow(self.img_chose, origin='lower', cmap="gnuplot", norm=colors.SymLogNorm(linthresh=self.thresh_chose), extent=[-arcsec_extent, arcsec_extent, -arcsec_extent, arcsec_extent])

        X_in, Y_in = self.Ellipse(R_in, pixelscale)
        X_out, Y_out = self.Ellipse(R_out, pixelscale)
        # ax_img_Extraction.fill(np.append(X_in, X_out[::-1]), np.append(Y_in, Y_out[::-1]), c='green', alpha=0.3, ls='')
        ax_img_Extraction.set_facecolor('black')
        ax_img_Extraction.scatter(X_in,  Y_in,  s=1, c='green')
        ax_img_Extraction.scatter(X_out, Y_out, s=1, c='green')
        ax_phF.errorbar(Scatt, PI, xerr=Err_Scatt, yerr=np.abs(Err_PI), marker='.', capsize=2, color='black', label='LB uncorrected', ls='dashed', alpha=0.2)
        # ax_phF.errorbar(Scatt, PI, xerr=Err_Scatt, yerr=np.abs(Err_PI), color='black', label='all disk')
        ax_phF.errorbar(Scatt, PI_LB,  xerr=Err_Scatt,  yerr=np.abs(Err_PI_LB), marker='.', capsize=2, color='black', label='LB corrected')
        # ax_phF.errorbar(Scatt_east, PI_east, yerr=np.abs(Err_PI_east), color='red',  label='east side', ls='dotted', alpha=0.3)
        # ax_phF.errorbar(Scatt_west, PI_west, yerr=np.abs(Err_PI_west), color='blue', label='west side', ls='dotted', alpha=0.3)
        if 'MCFOST' in self.file_name:
            MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, Err_MCFOST_I, Err_MCFOST_PI, Err_MCFOST_DoP = Tools.MCFOST_PhaseFunction('/'.join(self.file_path.split('/')[:-2]), self.file_name[:-5], self.normalization)
            NormMCFOSTPI = np.interp(90, MCFOST_Scatt, MCFOST_PI)
            MCFOST_PI = MCFOST_PI/NormMCFOSTPI
            ax_phF.errorbar(MCFOST_Scatt, np.abs(MCFOST_PI), yerr=np.abs(Err_MCFOST_PI), ls='dashed', alpha=0.5, color='purple', label='intrinsic')
        # Fig_img_phF.suptitle(self.disk_name, fontweight="bold")
        ax_phF.legend(loc='upper right')
        ax_phF.set_xlabel('Scattering angle [degree]')
        ax_phF.set_ylabel('Normalized Polarized Intensity')
        ax_phF.set_title("Normalized Polarized Phase Function")
        # ax_img.set_xlabel("$ \Delta $DEC (arcsec)")
        # ax_img.set_ylabel("$ \Delta $RA (arcsec)")
        ax_img_Extraction.set_xlabel("$ \Delta $DEC (arcsec)")
        ax_img_Extraction.set_ylabel("$ \Delta $RA (arcsec)")
        ax_img_Extraction.set_title("Polarized Intensity Image")
        # ax_img_Extraction.set_ylabel("pixel")
        layout_Img_PhF.addWidget(Toolbar)
        layout_Img_PhF.addWidget(Canvas_Img_PhF)
        self.Img_PhF.setLayout(layout_Img_PhF)
        Fig_img_phF.subplots_adjust(wspace=0.5)
        Fig_img_phF.tight_layout()
        self.Img_PhF.show()

    def Open_Header(self):
        options = QFileDialog.Options()
        with fits.open(self.file_path) as hdul:
            header = hdul[0].header
            header_text = repr(header)
            self.h_refeader_window = HeaderWindow(header_text)
            self.h_refeader_window.show()

class HeaderWindow(QWidget):
    def __init__(self, header_text):
        super().__init__()
        self.setWindowTitle('FITS Header')
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()
        
        self.text_edit = QTextEdit()
        self.text_edit.setText(header_text)
        self.text_edit.setReadOnly(True)
        
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = FileExplorerApp()
    ex.show()
    sys.exit(app.exec_())
