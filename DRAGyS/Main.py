import sys
import os
import pathlib
current_folder = pathlib.Path(__file__).parent
sys.path.append(current_folder)
import numpy as np
import time
from PyQt6.QtWidgets import QTextEdit, QApplication, QScrollArea, QDialog, QSpacerItem, QSizePolicy, QWidget, QSlider, QCheckBox, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QFrame, QProgressBar
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.patches import Wedge

from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib import colors
from matplotlib.figure import Figure

import pickle as pkl

from astropy.io import fits
import Tools as Tools
import SPF_Window as SPF
from   Filter_Pixel import FilteringWindow


class DRAGyS(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.normalization      = False
        self.LogScale           = False
        self.InputParams        = False
        self.InputStar          = False 
        self.Image_Displayed    = False
        self.Display_EZ         = False
        self.CheckEZ            = False
        self.fit_type = 'Polarized'
        self.img_type = 'Polarized'
        self.AzimuthalAngle = np.linspace(0, 359, 360)
        self.Fitting = None
        self.unit = 'Arcsec'
        self.List_Buttons    = []
        self.List_BigButtons = []
        self.List_Frame      = []
        self.GUI_Folder, self.Fitting_Folder, self.SPF_Folder = Tools.Init_Folders()

        # Files Browsers Buttons
        self.browse_button = QPushButton('Browse Files', self)
        self.browse_button.clicked.connect(self.browse_files)
        self.file_label = QLabel('No file selected', self)
        self.file_label.setFixedHeight(15)
        self.fit_file = QLabel("Is Fitting already done ? None", self)
        self.fit_file.setFixedHeight(15)

        # Fitting Buttons
        self.Display_Fit_button = QPushButton('Show Fitting', self)
        self.Display_Fit_button.clicked.connect(self.Show_Fitting)
        self.Display_Fit_button.setEnabled(False)
        self.Compute_Fit_button = QPushButton('Compute Fitting', self)
        self.Compute_Fit_button.clicked.connect(self.Launch_Filtering_Data)
        self.Compute_Fit_button.setEnabled(False)

        # Parameters Box
        self.UseFittingButton       = QPushButton("Polarized Fitting")
        self.UseFittingButton.clicked.connect(self.FittingType)
        self.UseFittingButton.setFixedWidth(80)
        self.Inclination_text       = QLabel('i = ', self)
        self.PositionAngle_text     = QLabel('PA = ', self)
        self.Scale_Height_text      = QLabel('h/r = ', self)
        self.Height_text            = QLabel('h = ', self)
        self.Radius_text            = QLabel('r = ', self)
        self.Alpha_text             = QLabel('alpha = ', self)
        self.Err_Inclination_text   = QLabel('° \u00B1 ', self)
        self.Err_PositionAngle_text = QLabel('° \u00B1 ', self)
        self.Err_Scale_Height_text  = QLabel(' \u00B1 ', self)
        self.Err_Height_text        = QLabel(' \u00B1 ', self)
        self.Err_Radius_text        = QLabel(' \u00B1 ', self)
        self.Err_Alpha_text         = QLabel(' \u00B1 ', self)
        self.InclinationLine        = QDoubleSpinBox(self)
        self.PositionAngleLine      = QDoubleSpinBox(self)
        self.AspectRatioLine        = QDoubleSpinBox(self)
        self.HeightLine             = QDoubleSpinBox(self)
        self.RadiusLine             = QDoubleSpinBox(self)
        self.PowerLawAlphaLine      = QDoubleSpinBox(self)
        self.ErrInclinationLine     = QDoubleSpinBox(self)
        self.ErrPositionAngleLine   = QDoubleSpinBox(self)
        self.ErrAspectRatioLine     = QDoubleSpinBox(self)
        self.ErrHeightLine          = QDoubleSpinBox(self)
        self.ErrRadiusLine          = QDoubleSpinBox(self)
        self.ErrPowerLawAlphaLine   = QDoubleSpinBox(self)

        self.InclinationLine.setRange(0.0, 10000.0)
        self.PositionAngleLine.setRange(0.0, 10000.0)
        self.AspectRatioLine.setRange(0.0, 10000.0)
        self.HeightLine.setRange(0.0, 10000.0)
        self.RadiusLine.setRange(0.0, 10000.0)
        self.PowerLawAlphaLine.setRange(0.0, 10000.0)
        self.ErrInclinationLine.setRange(0.0, 10000.0)
        self.ErrPositionAngleLine.setRange(0.0, 10000.0)
        self.ErrAspectRatioLine.setRange(0.0, 10000.0)
        self.ErrHeightLine.setRange(0.0, 10000.0)
        self.ErrRadiusLine.setRange(0.0, 10000.0)
        self.ErrPowerLawAlphaLine.setRange(0.0, 10000.0)

        self.InclinationLine.setDecimals(2)
        self.PositionAngleLine.setDecimals(2)
        self.AspectRatioLine.setDecimals(3)
        self.HeightLine.setDecimals(2)
        self.RadiusLine.setDecimals(2)
        self.PowerLawAlphaLine.setDecimals(2)
        self.ErrInclinationLine.setDecimals(2)
        self.ErrPositionAngleLine.setDecimals(2)
        self.ErrAspectRatioLine.setDecimals(3)
        self.ErrHeightLine.setDecimals(2)
        self.ErrRadiusLine.setDecimals(2)
        self.ErrPowerLawAlphaLine.setDecimals(2)
        
        self.InclinationLine.setValue(0)
        self.PositionAngleLine.setValue(0)
        self.AspectRatioLine.setValue(0)
        self.HeightLine.setValue(10)
        self.RadiusLine.setValue(100)
        self.PowerLawAlphaLine.setValue(1)
        self.ErrInclinationLine.setValue(0)
        self.ErrPositionAngleLine.setValue(0)
        self.ErrAspectRatioLine.setValue(0)
        self.ErrHeightLine.setValue(0)
        self.ErrRadiusLine.setValue(0)
        self.ErrPowerLawAlphaLine.setValue(0)

        self.InclinationLine.setFixedWidth(80)
        self.PositionAngleLine.setFixedWidth(80)
        self.AspectRatioLine.setFixedWidth(80)
        self.HeightLine.setFixedWidth(80)
        self.RadiusLine.setFixedWidth(80)
        self.PowerLawAlphaLine.setFixedWidth(80)
        self.ErrInclinationLine.setFixedWidth(80)
        self.ErrPositionAngleLine.setFixedWidth(80)
        self.ErrAspectRatioLine.setFixedWidth(80)
        self.ErrHeightLine.setFixedWidth(80)
        self.ErrRadiusLine.setFixedWidth(80)
        self.ErrPowerLawAlphaLine.setFixedWidth(80)

        self.InclinationLine.editingFinished.connect(self.UpdateParams)
        self.ErrInclinationLine.editingFinished.connect(self.UpdateParams)
        self.PositionAngleLine.editingFinished.connect(self.UpdateParams)
        self.ErrPositionAngleLine.editingFinished.connect(self.UpdateParams)
        self.RadiusLine.editingFinished.connect(self.UpdateParams)
        self.ErrRadiusLine.editingFinished.connect(self.UpdateParams)
        self.HeightLine.editingFinished.connect(self.UpdateParams)
        self.ErrHeightLine.editingFinished.connect(self.UpdateParams)
        self.PowerLawAlphaLine.editingFinished.connect(self.UpdateParams)
        self.ErrPowerLawAlphaLine.editingFinished.connect(self.UpdateParams)

        self.InclinationLine.valueChanged.connect(self.Extraction_Zone)
        self.PositionAngleLine.valueChanged.connect(self.Extraction_Zone)
        self.HeightLine.valueChanged.connect(self.Extraction_Zone)
        self.RadiusLine.valueChanged.connect(self.Extraction_Zone)
        self.PowerLawAlphaLine.valueChanged.connect(self.Extraction_Zone)
        
        # Star position buttons
        self.CheckStar = QCheckBox('Non Centered Star', self)
        self.X_StarPositionLabel = QLabel("x = ", self)
        self.Y_StarPositionLabel = QLabel("y = ", self)
        self.X_StarPosition = QDoubleSpinBox(self)
        self.Y_StarPosition = QDoubleSpinBox(self)
        self.X_StarPosition.setRange(0.0, 10000.0)
        self.Y_StarPosition.setRange(0.0, 10000.0)
        self.X_StarPosition.setFixedWidth(50)
        self.Y_StarPosition.setFixedWidth(50)
        self.CheckStar.stateChanged.connect(self.on_check_Star_Position)
        self.CheckStar.setChecked(False)
        self.CheckStar.setEnabled(False)
        self.X_StarPosition.setEnabled(False)
        self.Y_StarPosition.setEnabled(False)

        # Image View buttons
        self.img_0_button  = QPushButton('Image 1', self)
        self.img_1_button  = QPushButton('Image 2', self)
        self.img_2_button  = QPushButton('Image 3', self)
        self.img_3_button  = QPushButton('Image 4', self)
        self.img_4_button  = QPushButton('Image 5', self)
        self.img_5_button  = QPushButton('Image 6', self)
        self.HeaderButton  = QPushButton('Header',  self)
        self.AzimuthButton = QPushButton('Remove Azimuth',  self)

        self.img_0_button.clicked.connect(lambda: self.Change_View(0))
        self.img_1_button.clicked.connect(lambda: self.Change_View(1))
        self.img_2_button.clicked.connect(lambda: self.Change_View(2))
        self.img_3_button.clicked.connect(lambda: self.Change_View(3))
        self.img_4_button.clicked.connect(lambda: self.Change_View(4))
        self.img_5_button.clicked.connect(lambda: self.Change_View(5))

        self.HeaderButton.clicked.connect(self.Open_Header)
        self.AzimuthButton.clicked.connect(self.LaunchAzimuthRemover)
        self.img_0_button.setEnabled(False)
        self.img_1_button.setEnabled(False)
        self.img_2_button.setEnabled(False)
        self.img_3_button.setEnabled(False)
        self.img_4_button.setEnabled(False)
        self.img_5_button.setEnabled(False)
        self.HeaderButton.setEnabled(False)
        self.AzimuthButton.setEnabled(False)
        self.HeaderButton.setFixedHeight(80)
        self.AzimuthButton.setFixedHeight(40)

        # Observations Parameters Buttons
        self.diameter_label   = QLabel('Telescope Diameter (m) : ', self)
        self.wavelength_label = QLabel('Wavelength ($\mu$m) : ', self)
        self.pixelscale_label = QLabel('Pixelscale ("/pix) : ', self)
        self.distance_label   = QLabel('Distance (pc) : ', self)
        self.R_in_label       = QLabel('R_in (au) : ', self)
        self.R_out_label      = QLabel('R_out (au) : ', self)
        self.n_bin_label      = QLabel('nb bins : ', self)
        self.diameter_entry   = QDoubleSpinBox(self)
        self.wavelength_entry = QDoubleSpinBox(self)
        self.pixelscale_entry = QDoubleSpinBox(self)
        self.distance_entry   = QDoubleSpinBox(self)
        self.nb_bin_entry     = QSpinBox(self)
        self.diameter_entry.setRange(0.0, 10000.0)
        self.wavelength_entry.setRange(0.0, 10000.0)
        self.pixelscale_entry.setRange(0.0, 10000.0)
        self.distance_entry.setRange(0.0, 10000.0)
        self.diameter_entry.setDecimals(3)
        self.wavelength_entry.setDecimals(3)
        self.pixelscale_entry.setDecimals(7)
        self.distance_entry.setDecimals(3)
        # self.nb_bin_entry.setRange(0.0, 10000.0)
        self.diameter_entry.setValue(8.2)
        self.wavelength_entry.setValue(1.65)
        # self.pixelscale_entry.setEnabled(False)
        # self.distance_entry.setEnabled(False)
        # self.nb_bin_entry.setEnabled(False)
        # self.pixelscale_entry.setEnabled(False)
        self.diameter_entry.setFixedWidth(80)
        self.wavelength_entry.setFixedWidth(80)
        self.pixelscale_entry.setFixedWidth(80)
        self.distance_entry.setFixedWidth(80)
        self.nb_bin_entry.setFixedWidth(80)

        # Extraction zone Buttons
        self.R_in_entry  = QDoubleSpinBox(self)
        self.R_out_entry = QDoubleSpinBox(self)
        self.R_in_entry.setRange(0.0, 10000.0)
        self.R_out_entry.setRange(0.0, 10000.0)
        self.R_in_entry.setMinimum(0)
        self.R_in_entry.setMaximum(1000)
        self.R_out_entry.setMinimum(0)
        self.R_out_entry.setMaximum(1000)
        self.R_in_entry.setFixedWidth(100)
        self.R_out_entry.setFixedWidth(100)
        self.R_adjustement = QCheckBox('See Adjustment', self)
        self.R_adjustement.setEnabled(False)
        self.R_in_entry.valueChanged.connect(self.Extraction_Zone)
        self.R_out_entry.valueChanged.connect(self.Extraction_Zone)
        self.R_in_entry.lineEdit().returnPressed.connect(self.Extraction_Zone)
        self.R_out_entry.lineEdit().returnPressed.connect(self.Extraction_Zone)
        self.R_adjustement.stateChanged.connect(self.Ring_Adjust_2)

        # Zoom button
        self.ZoomLabel  = QLabel("Zoom : ", self)
        self.ZoomLabel.setFixedHeight(15)
        self.ZoomSlider = QSlider(Qt.Orientation.Horizontal)
        self.ZoomSlider.setMinimum(100)
        self.ZoomSlider.setMaximum(2000)
        self.ZoomSlider.setValue(1)
        self.ZoomSlider.setSingleStep(1)
        self.ZoomSlider.valueChanged.connect(self.Zoom_Slider_Update)

        # PhFs button
        self.SPF_Type_button      = QPushButton("Polarized")
        self.Compute_PhF_button   = QPushButton('Compute SPF', self)
        self.Show_disk_PhF_button = QPushButton('Show All SPF', self)
        self.Show_img_PhF_button  = QPushButton("Image & SPF", self)
        self.SPF_Type_button.setFixedHeight(30)
        # self.Compute_PhF_button.setEnabled(False)
        self.Show_disk_PhF_button.setEnabled(False)
        self.Show_img_PhF_button.setEnabled(False)
        self.SPF_Type_button.clicked.connect(self.Change_total_polarized)
        self.Compute_PhF_button.clicked.connect(self.Run_SPF)
        self.Show_disk_PhF_button.clicked.connect(self.Show_disk_PhaseFunction)
        self.Show_img_PhF_button.clicked.connect(self.Show_img_PhF)
        self.Is_computed = QLabel('None')
        self.Is_computed.setFixedHeight(15)

        # Add Button in List
        self.List_Buttons.append(self.browse_button)
        self.List_Buttons.append(self.Display_Fit_button)
        self.List_Buttons.append(self.Compute_Fit_button)
        self.List_Buttons.append(self.R_adjustement)
        self.List_Buttons.append(self.Compute_PhF_button)
        self.List_Buttons.append(self.Show_disk_PhF_button)
        self.List_Buttons.append(self.Show_img_PhF_button)
        
        self.List_Buttons.append(self.img_0_button)
        self.List_Buttons.append(self.img_1_button)
        self.List_Buttons.append(self.img_2_button)
        self.List_Buttons.append(self.img_3_button)
        self.List_Buttons.append(self.img_4_button)
        self.List_Buttons.append(self.img_5_button)
        self.List_Buttons.append(self.AzimuthButton)
        self.List_Buttons.append(self.HeaderButton)
        self.List_BigButtons.append(self.HeaderButton)

        for Button in self.List_Buttons:
            if Button in self.List_BigButtons:
                Button.setFixedHeight(90)
            else :
                Button.setFixedHeight(40)

        # Display Figure
        self.fig = Figure(facecolor="k")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        self.ax.set_ylabel('$\Delta$ RA (arcsec)')
        self.ax.set_xlabel('$\Delta$ DEC (arcsec)')
        self.ax.set_aspect('equal')
        self.ax.spines['bottom'].set_color('w')
        self.ax.spines['top'].set_color('w')
        self.ax.spines['left'].set_color('w')
        self.ax.spines['right'].set_color('w')
        self.ax.tick_params(axis='x', colors='w')
        self.ax.tick_params(axis='y', colors='w')
        self.ax.xaxis.label.set_color('w')
        self.ax.yaxis.label.set_color('w')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setFixedHeight(500)
        layout = QHBoxLayout()

        # Browse files
        Browsebox = QFrame()
        Browsebox.setFrameShape(QFrame.Shape.StyledPanel)
        Browsebox.setFrameShadow(QFrame.Shadow.Raised)
        browsebox = QVBoxLayout()
        browsebox.addWidget(self.browse_button)
        browsebox.addWidget(self.file_label)
        Browsebox.setLayout(browsebox)

        # Fitting
        FitBox = QFrame()
        FitBox.setFrameShape(QFrame.Shape.StyledPanel)
        FitBox.setFrameShadow(QFrame.Shadow.Raised)
        FitBoxButton = QVBoxLayout()
        FitBoxButton.addWidget(self.Display_Fit_button)
        FitBoxButton.addWidget(self.Compute_Fit_button)
        FitBoxButton.addWidget(self.fit_file)        

        Errorbox = QVBoxLayout()
        Errorbox.addWidget(self.ErrInclinationLine)
        Errorbox.addWidget(self.ErrPositionAngleLine)
        Errorbox.addWidget(self.ErrHeightLine)
        Errorbox.addWidget(self.ErrRadiusLine)
        Errorbox.addWidget(self.ErrAspectRatioLine)
        Errorbox.addWidget(self.ErrPowerLawAlphaLine)
        
        PMBox = QVBoxLayout()
        PMBox.addWidget(self.Err_Inclination_text)
        PMBox.addWidget(self.Err_PositionAngle_text)
        PMBox.addWidget(self.Err_Height_text)
        PMBox.addWidget(self.Err_Radius_text)
        PMBox.addWidget(self.Err_Scale_Height_text)
        PMBox.addWidget(self.Err_Alpha_text)

        ValueBox = QVBoxLayout()
        ValueBox.addWidget(self.InclinationLine)
        ValueBox.addWidget(self.PositionAngleLine)
        ValueBox.addWidget(self.HeightLine)
        ValueBox.addWidget(self.RadiusLine)
        ValueBox.addWidget(self.AspectRatioLine)
        ValueBox.addWidget(self.PowerLawAlphaLine)

        TxtBox = QVBoxLayout()
        TxtBox.addWidget(self.Inclination_text)
        TxtBox.addWidget(self.PositionAngle_text)
        TxtBox.addWidget(self.Height_text)
        TxtBox.addWidget(self.Radius_text)
        TxtBox.addWidget(self.Scale_Height_text)
        TxtBox.addWidget(self.Alpha_text)

        StarPositionBox = QVBoxLayout()
        StarPositionBox.addWidget(self.CheckStar)
        xPosition = QHBoxLayout()
        yPosition = QHBoxLayout()
        xPosition.addWidget(self.X_StarPositionLabel)
        xPosition.addWidget(self.X_StarPosition)
        yPosition.addWidget(self.Y_StarPositionLabel)
        yPosition.addWidget(self.Y_StarPosition)
        xPosition.setAlignment(self.X_StarPositionLabel, Qt.AlignmentFlag.AlignRight)
        yPosition.setAlignment(self.Y_StarPositionLabel, Qt.AlignmentFlag.AlignRight)
        xPosition.setAlignment(self.X_StarPosition, Qt.AlignmentFlag.AlignLeft)
        yPosition.setAlignment(self.Y_StarPosition, Qt.AlignmentFlag.AlignLeft)
        StarPositionBox.addLayout(xPosition)
        StarPositionBox.addLayout(yPosition)
        FitBoxButton.addLayout(StarPositionBox)

        VFitParameters = QVBoxLayout()
        FitParameters = QHBoxLayout()
        FitParameters.setAlignment(Qt.AlignmentFlag.AlignRight)
        FitParameters.addLayout(TxtBox)
        FitParameters.addLayout(ValueBox)
        FitParameters.addLayout(PMBox)
        FitParameters.addLayout(Errorbox)
        VFitParameters.addLayout(FitParameters)
        VFitParameters.addWidget(self.AzimuthButton)
        FitParameters.setAlignment(TxtBox,   Qt.AlignmentFlag.AlignRight)
        FitParameters.setAlignment(ValueBox, Qt.AlignmentFlag.AlignRight)
        FitParameters.setAlignment(PMBox,    Qt.AlignmentFlag.AlignRight)
        FitParameters.setAlignment(Errorbox, Qt.AlignmentFlag.AlignRight)

        fitbox0 = QHBoxLayout()
        fitbox0.addLayout(FitBoxButton)
        spacer = QSpacerItem(50, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        fitbox0.addItem(spacer)
        fitbox0.addLayout(VFitParameters)

        fitbox = QVBoxLayout()
        # fitbox.addWidget(self.CheckInputParams)
        fitbox.addWidget(self.UseFittingButton)
        fitbox.addLayout(fitbox0)
        FitBox.setLayout(fitbox)
        fitbox.setAlignment(self.AzimuthButton, Qt.AlignmentFlag.AlignRight)
        # fitbox.setAlignment(self.CheckInputParams, Qt.AlignmentFlag.AlignRight)
        fitbox.setAlignment(self.UseFittingButton, Qt.AlignmentFlag.AlignRight)
    
        # Parameters
        diameterbox = QHBoxLayout()
        diameterbox.addWidget(self.diameter_label)
        diameterbox.addWidget(self.diameter_entry)
        wavelengthbox = QHBoxLayout()
        wavelengthbox.addWidget(self.wavelength_label)
        wavelengthbox.addWidget(self.wavelength_entry)
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
        LeftParaBox.addLayout(diameterbox)
        LeftParaBox.addLayout(wavelengthbox)
        LeftParaBox.addLayout(pixelscalebox)
        LeftParaBox.addLayout(Distancebox)
        LeftParaBox.addLayout(nb_binbox)

        RightParaBox = QVBoxLayout()
        RightParaBox.addLayout(R_inbox)
        RightParaBox.addLayout(R_outbox)
        RightParaBox.addWidget(self.R_adjustement)

        ParaBox = QFrame()
        ParaBox.setFrameShape(QFrame.Shape.StyledPanel)
        ParaBox.setFrameShadow(QFrame.Shadow.Raised)
        parabox = QHBoxLayout()
        parabox.addLayout(LeftParaBox)
        parabox.addLayout(RightParaBox)
        ParaBox.setLayout(parabox)

        # Phase Function
        PhFbox = QVBoxLayout()
        PhFBox_button = QHBoxLayout()
        PhFBox_button.addWidget(self.Compute_PhF_button)
        PhFBox_button.addWidget(self.Show_disk_PhF_button)
        PhFBox_button.addWidget(self.Show_img_PhF_button)

        PhFbox.addWidget(self.SPF_Type_button)
        PhFbox.addLayout(PhFBox_button)
        PhFbox.addWidget(self.Is_computed)

        AllPhFbox = QFrame()
        AllPhFbox.setFrameShape(QFrame.Shape.StyledPanel)
        AllPhFbox.setFrameShadow(QFrame.Shadow.Raised)

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

        # Imshow
        DisplayBox = QFrame()
        DisplayBox.setFixedHeight(150)
        DisplayBox.setFrameShape(QFrame.Shape.StyledPanel)
        DisplayBox.setFrameShadow(QFrame.Shadow.Raised)
        displaybox  = QVBoxLayout()
        displaybox1 = QHBoxLayout()
        displaybox1.addLayout(h1box)
        displaybox1.addLayout(h2box)
        displaybox1.addLayout(h3box)
        displaybox1.addLayout(h4box)
        displaybox.addLayout(displaybox1)
        DisplayBox.setLayout(displaybox)

        # Organisation
        LeftBBox = QFrame()
        LeftBBox.setFixedWidth(500)
        LeftBBox.setFrameShape(QFrame.Shape.StyledPanel)
        LeftBBox.setFrameShadow(QFrame.Shadow.Raised)
        leftbbox = QVBoxLayout()
        leftbbox.addWidget(Browsebox)
        leftbbox.addWidget(FitBox)
        leftbbox.addWidget(ParaBox)
        leftbbox.addWidget(AllPhFbox)
        LeftBBox.setLayout(leftbbox)
        
        RightBBox = QFrame()
        RightBBox.setFrameShape(QFrame.Shape.StyledPanel)
        RightBBox.setFrameShadow(QFrame.Shadow.Raised)
        rightbbox = QVBoxLayout()
        rightbbox.addWidget(self.ZoomLabel)
        rightbbox.addWidget(self.ZoomSlider)
        rightbbox.addWidget(self.canvas)
        rightbbox.setAlignment(self.canvas, Qt.AlignmentFlag.AlignCenter)
        rightbbox.addWidget(DisplayBox)
        RightBBox.setLayout(rightbbox)

        layout.addWidget(LeftBBox)
        layout.addWidget(RightBBox)
        self.setLayout(layout)        
        self.setGeometry(50, 50, 1200, 500)
        self.setWindowTitle('GUI Disk Fitting Yields to Scattering Phase Function Extraction')
            
    def Set_Initial_Values(self):
        '''
        Difinition of parameters with respect to what it's return on data image name or headers
        '''
        self.nb_bin_entry.setEnabled(True)
        self.pixelscale_entry.setEnabled(True)
        self.R_in_entry.setEnabled(True)
        self.R_out_entry.setEnabled(True)
        self.distance_entry.setEnabled(True)
        self.nb_bin_entry.setValue(37)  # To have 5° width bin size
        self.pixelscale_entry.setValue(0.01225)
        try :
            Distances = Tools.DiskDistances()
            self.distance_entry.setValue(str(Tools.Get_Distance(Distances, self.file_path)))
        except:
            self.distance_entry.setValue(100) 
        if "SPHERE" in self.file_name.upper():
            self.pixelscale_entry.setValue(0.01225)

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
            self.R_in_entry.setValue(175)
            self.R_out_entry.setValue(185)
            self.pixelscale_entry.setValue(0.01225)

        if 'MCFOST_Si_a_' in self.file_name:
            self.R_in_entry.setValue(180)
            self.R_out_entry.setValue(280)
            self.pixelscale_entry.setValue(0.01996007982)

        if 'MCFOST_AmCa_a_' in self.file_name:
            self.R_in_entry.setValue(180)
            self.R_out_entry.setValue(280)
            self.pixelscale_entry.setValue(0.01996007982)
        
        if 'VBB' in self.file_name:
            self.pixelscale_entry.setValue(0.0036)
        self.pixelscale = float(self.pixelscale_entry.value())

# ========================================================================================
# ================================    File Finder    =====================================
# ========================================================================================

    def browse_files(self):
        self.R_adjustement.setEnabled(True)
        self.R_adjustement.setChecked(False)
        self.CheckEZ = False
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
            # self.fit_file_name = 'Fitting_' + self.disk_name + '_' + self.img_type + '.pkl'
            self.fit_file_name = f'Fitting_{self.disk_name}_{self.fit_type}.pkl'
            self.Set_Initial_Values()
            self.Init_image(self.file_path)
            self.SavedParams()
        if self.Fitting and self.Fitting.isVisible():
            self.Fitting.close()

# ==================================================================
# =====================    Image Display   =========================
# ==================================================================
    def Change_View(self, img_show):
        self.img_chose    = self.all_img[img_show]
        self.thresh_chose = self.all_thresh[img_show]
        try:
            self.displayed_img.remove()
        except:
            SERGE = "Scattering Extraction from Ring Geometry Estimation"
        # self.displayed_img = self.ax.imshow(self.img_chose, extent=[-self.Size, self.Size, -self.Size, self.Size], origin='lower', cmap="gnuplot", norm=colors.SymLogNorm(linthresh=self.thresh_chose))
        self.displayed_img = self.ax.imshow(self.img_chose, extent=[-self.Size, self.Size, -self.Size, self.Size], origin='lower', cmap="gnuplot", norm=colors.LogNorm())
        self.Zoom_Slider_Update(self.ZoomSlider.value())

    def Init_image(self, file_path):
        [self.img_0,    self.img_1,    self.img_2,    self.img_3,    self.img_4,    self.img_5], [self.thresh_0, self.thresh_1, self.thresh_2, self.thresh_3, self.thresh_4, self.thresh_5] = Tools.Images_Opener(file_path)
        PixelScale      = float(self.pixelscale_entry.value())
        self.Size       = len(self.img_0)/2 * PixelScale
        self.all_img    = [self.img_0,    self.img_1,    self.img_2,    self.img_3,    self.img_4,    self.img_5]
        self.all_thresh = [self.thresh_0, self.thresh_1, self.thresh_2, self.thresh_3, self.thresh_4, self.thresh_5]
        self.all_name   = ["IMG0",  "IMG1", "IMG2",  "IMG3", "IMG4",  "IMG5"]

        self.img_0_button.setEnabled(True)
        self.img_1_button.setEnabled(True)
        self.img_2_button.setEnabled(True)
        self.img_3_button.setEnabled(True)
        self.img_4_button.setEnabled(True)
        self.img_5_button.setEnabled(True)
        self.HeaderButton.setEnabled(True)
        self.AzimuthButton.setEnabled(True)

        self.Change_View(0)
        self.Zoom_Slider_Update(self.ZoomSlider.value())
        self.CheckStar.setEnabled(True)
        self.X_StarPosition.setValue(float(len(self.img_0)/2))
        self.Y_StarPosition.setValue(float(len(self.img_0)/2))

    def Zoom_Slider_Update(self, value):
        if self.Image_Displayed:
            PixelScale = float(self.pixelscale_entry.value())
            size  = len(self.img_0) * 100/value
            x_min = len(self.img_0)/2 - size/2
            x_max = len(self.img_0)/2 + size/2
            self.ax.set_xlim((x_min - len(self.img_0)/2)* PixelScale, (x_max - len(self.img_0)/2)* PixelScale)
            self.ax.set_ylim((x_min - len(self.img_0)/2)* PixelScale, (x_max - len(self.img_0)/2)* PixelScale)
            self.canvas.draw()
    
# ==================================================================
# =====================    Fitting Part   ==========================
# ==================================================================
    def FittingType(self):
        """
        When click on "Polarized/Total Fitting", to update displayed parameters
        """
        if self.fit_type   == 'Total':
            self.fit_type  =  'Polarized'
            self.UseFittingButton.setText("Polarized Fitting")
        elif self.fit_type == 'Polarized':
            self.fit_type  =  'Total'
            self.UseFittingButton.setText("Total Fitting")
        self.SavedParams()

    def SavedParams(self):
        """ 
        When open a new disk file
        When compute fitting is done
        When click on "Polarized/Total Fitting", to update displayed parameters
        """
        # Fit_Name = f'Fitting_{self.disk_name}_{self.fit_type}.pkl'
        try :
            with open(f'{self.Fitting_Folder}/Fitting_{self.disk_name}_{self.fit_type}.pkl', 'rb') as fichier:
                Loaded_Data = pkl.load(fichier)
            [self.incl,   self.r_ref,   self.h_ref,   self.aspect,   self.alpha,   self.PA]   = Loaded_Data['params']
            [self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, self.D_alpha, self.D_PA] = Loaded_Data['Err']
            self.UpdateFitting()
        except :
            print("No fitting file found")
            self.UpdateParams()

    def UpdateParams(self):
        """
        When one parameter is modified
        """
        self.incl     = np.radians(float(self.InclinationLine.value()))
        self.D_incl   = np.radians(float(self.ErrInclinationLine.value()))
        self.PA       = np.radians(float(self.PositionAngleLine.value()))
        self.D_PA     = np.radians(float(self.ErrPositionAngleLine.value()))
        self.r_ref    = float(self.RadiusLine.value())
        self.D_r_ref  = float(self.ErrRadiusLine.value())
        self.h_ref    = float(self.HeightLine.value())
        self.D_h_ref  = float(self.ErrHeightLine.value())
        self.alpha    = float(self.PowerLawAlphaLine.value())
        self.D_alpha  = float(self.ErrPowerLawAlphaLine.value())
        if self.r_ref == 0 or self.h_ref == 0:
            self.h_ref = 1e-20
            self.r_ref = 1e-20
        self.aspect   = self.h_ref/self.r_ref**self.alpha
        self.D_aspect = (self.h_ref/self.r_ref**self.alpha) * np.sqrt((self.D_r_ref/self.r_ref)**2 + (self.D_h_ref/self.h_ref)**2)
        self.UpdateFitting()

    def UpdateFitting(self):
        """
        When open a new disk file
        When compute fitting is done
        When click on "Polarized/Total Fitting", to update displayed parameters 
        """
        self.InclinationLine.setValue(np.round(np.degrees(self.incl),2))
        self.ErrInclinationLine.setValue(np.round(np.degrees(self.D_incl),2))
        self.PositionAngleLine.setValue(np.round(np.degrees(self.PA),2))
        self.ErrPositionAngleLine.setValue(np.round(np.degrees(self.D_PA),2))
        self.HeightLine.setValue(np.round(self.h_ref,3))
        self.ErrHeightLine.setValue(np.round(self.D_h_ref,3))
        self.RadiusLine.setValue(np.round(self.r_ref,3))
        self.ErrRadiusLine.setValue(np.round(self.D_r_ref,3))
        self.AspectRatioLine.setValue(np.round(self.aspect,3))
        self.ErrAspectRatioLine.setValue(np.round(self.D_aspect,3))
        self.PowerLawAlphaLine.setValue(np.round(self.alpha, 3))
        self.ErrPowerLawAlphaLine.setValue(np.round(self.D_alpha, 3))

    def on_check_Star_Position(self, state):
        if state ==  2:
            self.InputStar = True
            self.X_StarPosition.setEnabled(True)
            self.Y_StarPosition.setEnabled(True)
        else:
            self.InputStar = False
            self.X_StarPosition.setValue(len(self.img_0)/2)
            self.Y_StarPosition.setValue(len(self.img_0)/2)
            self.X_StarPosition.setEnabled(False)
            self.Y_StarPosition.setEnabled(False)

    def Show_Fitting(self):
        if not self.Fitting or not self.Fitting.isVisible():
            try:
                self.Fitting_figure.clear()  # Nettoyer la figure
            except:
                first_code_name = 'MaxSoft'
            self.Fitting        = QWidget()
            self.Fitting_figure, ax = plt.subplots(1, 1)
            canvas_fit          = FigureCanvas(self.Fitting_figure)
            self.Fitting.setWindowTitle("Fitting Figure")
            (x_min, x_max) = self.ax.get_xlim()
            PixelScale     = float(self.pixelscale_entry.value())
            size           = len(self.img_0)/2 * PixelScale
            center         = len(self.img_0)/2
            
            layout_fit = QVBoxLayout()
            layout_fit.addWidget(canvas_fit)
            self.Fitting.setLayout(layout_fit)

            [Xc, Yc, X_e, Y_e, a, b, e] = Tools.Load_Structure(self.fit_file_name, Type='Ellipse')
            [X, Y, all_X, all_X] = Tools.Load_Structure(self.fit_file_name, Type='Points')
            ax.set_title("Numerically Stable Direct \n Least Squares Fitting of Ellipses", fontweight='bold')
            # ax.imshow(self.img_chose, origin='lower', extent=[-size, size, -size, size], cmap="gnuplot", norm=colors.SymLogNorm(linthresh=self.thresh_chose))
            ax.imshow(self.img_chose, origin='lower', extent=[-size, size, -size, size], cmap="gnuplot", norm=colors.LogNorm())
            ax.scatter((Y-center)*PixelScale, (X-center)*PixelScale, marker='.', color='blue') # given points
            ax.plot((Y_e-center)*PixelScale,  (X_e-center)*PixelScale, label="ellipse fit", color='blue')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ax.set_xlabel('X position [pix]')
            ax.set_ylabel('Y position [pix]')
            ax.legend(loc='upper right')
            ax.text(x_min, x_max, u'i = {:<15} \n PA = {:<15} \n h/r = {:<15}'.format(str(np.round(np.degrees(self.incl),3))+' °', str(np.round(np.degrees(self.PA),3))+' °', str(np.round(self.aspect, 3))), color='red', ha='left', va='top')
            self.Fitting.show()

# ==================================================================
# ===================    Data Filtering    =========================
# ==================================================================
    def Launch_Filtering_Data(self):
        (x_min, x_max) = self.ax.get_xlim() 
        PixelScale = float(self.pixelscale_entry.value())
        x_min, x_max = x_min/PixelScale + len(self.img_0)/2, x_max/PixelScale + len(self.img_0)/2
        D = float(self.diameter_entry.value())
        W = float(self.wavelength_entry.value())
        r_beam = np.degrees((W*1e-6)/D) *3600/PixelScale
        self.Filtering_Window = FilteringWindow(self.disk_name, self.fit_type, self.img_chose, self.thresh_chose, x_min, x_max, r_beam)
        self.Filtering_Window.exec()
        if self.fit_file_name in os.listdir(self.Fitting_Folder):
            self.Display_Fit_button.setEnabled(True)
            self.fit_file.setText("Is Fitting already? Yes")
            self.fit_file.setStyleSheet('color: green;')
            self.SavedParams()

# ==================================================================
# ===================    Extraction Zone   =========================
# ==================================================================
    def Ring_Adjust_2(self, state):
        if state == 0:
            self.CheckEZ = False
            self.R_adjustement.setChecked(False)
            if self.Display_EZ:
                self.ellipse_in.remove()
                for fill_obj in self.ellipse_zone:
                    fill_obj.remove()
                self.ellipse_out.remove()
                self.canvas.draw()
                self.Display_EZ = False
        else:
            self.CheckEZ = True
            self.R_adjustement.setChecked(True)
            self.Extraction_Zone()
            
    def Extraction_Zone(self):
        if self.CheckEZ:
            if self.Display_EZ:
                self.ellipse_in.remove()
                self.ellipse_out.remove()
                for fill_obj in self.ellipse_zone:
                    fill_obj.remove()
                self.Display_EZ = False
            R_in          = float(self.R_in_entry.value())
            R_out         = float(self.R_out_entry.value())
            pixelscale    = float(self.pixelscale_entry.value())
            X_in, Y_in    = self.Ellipse(R_in, pixelscale) #*1.996007984)
            X_out, Y_out  = self.Ellipse(R_out, pixelscale) #*1.996007984)
            self.ellipse_zone = self.ax.fill(np.append(X_in, X_out[::-1]), 
                                    np.append(Y_in, Y_out[::-1]), 
                                    color='gold', alpha=0.4, linestyle='')
            self.ellipse_in   = self.ax.scatter(X_in, Y_in, c='orange', s=1)
            self.ellipse_out  = self.ax.scatter(X_out, Y_out, c='orange', s=1)
            self.canvas.draw()
            self.Display_EZ = True

    def Ellipse(self, R, pixelscale):
        [Xc, Yc, _, _, _, _, _] = Tools.Load_Structure(f'Fitting_{self.disk_name}_{self.fit_type}.pkl', Type='Ellipse')
        xs = ys = len(self.img_0)/2
        yc_p, xc_p = Tools.Orthogonal_Prejection((xs, ys), (Yc, Xc), np.pi/2 - self.PA)
        Phi = np.radians(np.linspace(0, 359, 360))
        d = float(self.distance_entry.value())
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

    def Change_total_polarized(self):
        if self.SPF_Type_button.text() =='Polarized':
            self.SPF_Type_button.setText('Total')
            self.img_type = 'Total'
        else:
            self.SPF_Type_button.setText("Polarized")
            self.img_type = "Polarized"

        if self.img_type + '_' + self.disk_name + '.spf' in os.listdir(self.SPF_Folder) :
            self.Is_computed.setText(self.disk_name + " Phase Function is already computed")
            self.Is_computed.setStyleSheet('color: green')
            self.Show_disk_PhF_button.setEnabled(True)
            self.Show_img_PhF_button.setEnabled(True)
        else : 
            self.Is_computed.setText(self.disk_name + " Phase Function is not computed")
            self.Is_computed.setStyleSheet('color: red')
            self.Show_disk_PhF_button.setEnabled(False)
            self.Show_img_PhF_button.setEnabled(False)
        if self.Fitting and self.Fitting.isVisible():
            self.Fitting.close()

    def Run_SPF(self):
        distance   = float(self.distance_entry.value())
        R_in       = float(self.R_in_entry.value())
        R_out      = float(self.R_out_entry.value())
        n_bin      = int(self.nb_bin_entry.value())
        pixelscale = float(self.pixelscale_entry.value())
        D          = float(self.diameter_entry.value())
        W          = float(self.wavelength_entry.value())
        r_beam     = np.degrees((W*1e-6)/D) *3600/pixelscale
        self.UpdateParams()
        xs = float(self.X_StarPosition.value())
        ys = float(self.Y_StarPosition.value())
        if self.InputStar:
            [Xc, Yc, _, _, _, _, _] = Tools.Load_Structure(f'Fitting_{self.disk_name}_{self.fit_type}.pkl', Type='Ellipse')
        else:
            Xc, Yc = None, None

        start = time.time()
        self.total_steps = (2 * 300*360 + 3*len(self.img_0)*len(self.img_0) + 2*n_bin) * 7
        self.computation_step = 0
        list_params_SPF = [ [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA,             self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, "Total"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl - self.D_incl, self.PA,             self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, "Incl"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl + self.D_incl, self.PA,             self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, "Incl"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA - self.D_PA, self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, "PA"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA + self.D_PA, self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, "PA"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA,             self.r_ref, self.h_ref, self.aspect - self.D_aspect,    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, "Aspect"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA,             self.r_ref, self.h_ref, self.aspect + self.D_aspect,    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, "Aspect"]]
        SPF.Compute_SPF(list_params_SPF, self.file_name, self.img_type)
        end = time.time()
        self.Is_computed.setText(" SPF computed - time = " +str(np.round(end-start, 2)) + ' seconds')
        self.Is_computed.setStyleSheet('color: green')
        self.Show_disk_PhF_button.setEnabled(True)
        self.Show_img_PhF_button.setEnabled(True)

# ==================================================================
# ===================    Display Results   =========================
# ==================================================================

    def Show_disk_PhaseFunction(self):
        self.NormType    = '90'
        self.LBCorrected = True
        self.ShowSide    = True
        self.ShowMCFOST  = False
        self.Disk_PhF    = QWidget()
        self.Disk_PhF.setWindowTitle("Phase Functions")
        try:
            figure_Disk_PhF.clear()  # Nettoyer la figure
        except:
            Tool_Could_Also_be_named = 'DRAGyS'
        # figure_Disk_PhF = Figure()
        figure_Disk_PhF, (self.ax_Disk_PhF_I, self.ax_Disk_PhF_PI, self.ax_Disk_DoP) = plt.subplots(1, 3)
        self.ax_Disk_DoP.clear()
        self.ax_Disk_PhF_I.clear()
        self.ax_Disk_PhF_PI.clear()
        self.canvas_Disk_PhF = FigureCanvas(figure_Disk_PhF)
        Toolbar = NavigationToolbar(self.canvas_Disk_PhF, self)
        layout_Disk_PhF = QVBoxLayout()


        self.NormButton = QCheckBox("Normalize")
        LogButton       = QCheckBox("LogScale")
        LB_Remove       = QCheckBox("Limb Brightening Corrected")
        SideButton      = QCheckBox("Each Side")
        MCFOSTButton    = QCheckBox("MCFOST")

        LB_Remove.setChecked(True)
        SideButton.setChecked(True)

        if 'MCFOST' not in self.file_name:
            MCFOSTButton.setEnabled(False)
        MCFOSTButton.stateChanged.connect(self.MCFOSTCheck)
        self.NormButton.stateChanged.connect(self.NormalizeCheck)

        LogButton.stateChanged.connect(self.LogCheck)
        LB_Remove.stateChanged.connect(self.LBCheck)
        SideButton.stateChanged.connect(self.SideCheck)
        self.Update_SPF()

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
        if state == 2:
            self.NormType = '90'
            self.ax_Disk_PhF_I.autoscale()
        else:
            self.NormType = 'None'
            self.ax_Disk_PhF_I.autoscale()
        self.Update_SPF()
    
    def LBCheck(self, state):
        if state == 2:
            self.LBCorrected = True
        else:
            self.LBCorrected = False
        self.Update_SPF()
        
    def SideCheck(self, state):
        if state == 2:
            self.ShowSide = True
        else:
            self.ShowSide = False
        self.Update_SPF()
    
    def LogCheck(self, state):
        if state == 2:
            self.ax_Disk_PhF_I.set_yscale('log')
            self.ax_Disk_PhF_PI.set_yscale('log')
        else:
            self.ax_Disk_PhF_I.set_yscale('linear')
            self.ax_Disk_PhF_PI.set_yscale('linear')
        self.Update_SPF()

    def MCFOSTCheck(self, state):
        if state == 2:
            self.ShowMCFOST = True
            self.ShowNormalized = True
            self.NormButton.setChecked(True)
            self.NormButton.setEnabled(False)
            self.ax_Disk_PhF_I.autoscale()
        else:
            self.ShowMCFOST = False
            self.NormButton.setEnabled(True)
        self.Update_SPF()

    def Update_SPF(self):
        self.ax_Disk_PhF_I.cla()
        self.ax_Disk_PhF_PI.cla()
        self.ax_Disk_DoP.cla()
        self.ax_Disk_DoP.axvline(90, c='k', ls='dotted', alpha=0.3)
        self.ax_Disk_DoP.set_xlim(10, 170)
        self.ax_Disk_PhF_I.set_xlabel("Total Flux")
        self.ax_Disk_PhF_I.set_ylabel("Scattering Angle (deg)")
        self.ax_Disk_PhF_PI.set_xlabel("Polarized Flux")
        self.ax_Disk_PhF_PI.set_ylabel("Scattering Angle (deg)")
        self.ax_Disk_DoP.set_xlabel("Degree of Polarization")
        self.ax_Disk_DoP.set_ylabel("Scattering Angle (deg)")

        total_folder     = f"{self.SPF_Folder}/Total_{self.disk_name}.spf"
        polarized_folder = f"{self.SPF_Folder}/Polarized_{self.disk_name}.spf"
        try :
            self.atSca, _, self.atSPF, self.D_atSca, _, self.D_atSPF, self.atLB, self.D_atLB = Tools.Get_PhF(total_folder, side='All',  LBCorrected=self.LBCorrected, norm=self.NormType)
            self.etSca, _, self.etSPF, self.D_etSca, _, self.D_etSPF, self.etLB, self.D_etLB = Tools.Get_PhF(total_folder, side='East', LBCorrected=self.LBCorrected, norm=self.NormType)
            self.wtSca, _, self.wtSPF, self.D_wtSca, _, self.D_wtSPF, self.wtLB, self.D_wtLB = Tools.Get_PhF(total_folder, side='West', LBCorrected=self.LBCorrected, norm=self.NormType)
            self.TSPF = True
        except:
            self.TSPF = False
        try:
            self.apSca, _, self.apSPF, self.D_apSca, _, self.D_apSPF, self.apLB, self.D_apLB = Tools.Get_PhF(polarized_folder, side='All',  LBCorrected=self.LBCorrected, norm=self.NormType)
            self.epSca, _, self.epSPF, self.D_epSca, _, self.D_epSPF, self.epLB, self.D_epLB = Tools.Get_PhF(polarized_folder, side='East', LBCorrected=self.LBCorrected, norm=self.NormType)
            self.wpSca, _, self.wpSPF, self.D_wpSca, _, self.D_wpSPF, self.wpLB, self.D_wpLB = Tools.Get_PhF(polarized_folder, side='West', LBCorrected=self.LBCorrected, norm=self.NormType)
            self.PSPF = True
        except:
            self.PSPF = False

        if self.TSPF and self.PSPF:
            self.adSca = np.linspace(np.max([np.min(self.atSca), np.min(self.apSca)]), np.min([np.max(self.atSca), np.max(self.apSca)]), 100)
            self.edSca = np.linspace(np.max([np.min(self.etSca), np.min(self.epSca)]), np.min([np.max(self.etSca), np.max(self.epSca)]), 100)
            self.wdSca = np.linspace(np.max([np.min(self.wtSca), np.min(self.wpSca)]), np.min([np.max(self.wtSca), np.max(self.wpSca)]), 100)
                    
            self.aDoP = np.interp(self.adSca, self.apSca, self.apSPF) / np.interp(self.adSca, self.atSca, self.atSPF)
            self.eDoP = np.interp(self.edSca, self.epSca, self.epSPF) / np.interp(self.edSca, self.etSca, self.etSPF)
            self.wDoP = np.interp(self.wdSca, self.wpSca, self.wpSPF) / np.interp(self.wdSca, self.wtSca, self.wtSPF)
            self.D_aDoP = self.D_eDoP = self.D_wDoP = 0

        if self.ShowMCFOST:
            MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, MCFOST_Err_I, MCFOST_Err_PI, MCFOST_Err_DoP = Tools.MCFOST_PhaseFunction('/'.join(self.file_path.split('/')[:-2]), self.file_name[:-5], True)
            if self.TSPF:
                self.I_MCFOST_Displayed   = self.ax_Disk_PhF_I.errorbar( MCFOST_Scatt, np.abs(MCFOST_I),   color='purple', label='MCFOST', alpha=0.4, ls='dashed')
            if self.PSPF:
                self.PI_MCFOST_Displayed  = self.ax_Disk_PhF_PI.errorbar(MCFOST_Scatt, np.abs(MCFOST_PI),  color='purple', label='MCFOST', alpha=0.4, ls='dashed')
            if self.TSPF and self.PSPF:
                self.DoP_MCFOST_Displayed = self.ax_Disk_DoP.errorbar(   MCFOST_Scatt, np.abs(MCFOST_DoP), color='purple', label='MCFOST', alpha=0.4, ls='dashed')
            self.MCFOST_Displayed = True

        if self.ShowMCFOST and self.NormType == 'None':
            normaT = np.interp(90, MCFOST_Scatt, MCFOST_I) / np.interp(90, self.atSca, self.atSPF)
            normeT = np.interp(90, MCFOST_Scatt, MCFOST_I) / np.interp(90, self.etSca, self.etSPF)
            normwT = np.interp(90, MCFOST_Scatt, MCFOST_I) / np.interp(90, self.wtSca, self.wtSPF)
            normaP = np.interp(90, MCFOST_Scatt, MCFOST_I) / np.interp(90, self.apSca, self.apSPF)
            normeP = np.interp(90, MCFOST_Scatt, MCFOST_I) / np.interp(90, self.epSca, self.epSPF)
            normwP = np.interp(90, MCFOST_Scatt, MCFOST_I) / np.interp(90, self.wpSca, self.wpSPF)
        else :
            normaT = normeT = normwT = normaP = normeP = normwP = 1

        marge = 0.1
        if self.TSPF :
            self.I_Displayed             = self.ax_Disk_PhF_I.errorbar( self.atSca, self.atSPF/normaT, xerr=self.D_atSca, yerr=np.abs(self.D_atSPF)/normaT, color='black', label='all disk')
            if self.ShowSide:
                self.I_east_Displayed    = self.ax_Disk_PhF_I.errorbar( self.etSca, self.etSPF/normeT, xerr=self.D_etSca, yerr=np.abs(self.D_etSPF)/normeT, color='blue',  label='east side', alpha=0.4, ls='dotted')
                self.I_west_Displayed    = self.ax_Disk_PhF_I.errorbar( self.wtSca, self.wtSPF/normwT, xerr=self.D_wtSca, yerr=np.abs(self.D_wtSPF)/normwT, color='red',   label='west side', alpha=0.4, ls='dotted')
                I_bound                  = np.concatenate([line.get_children()[0].get_ydata() for line in [self.I_Displayed,   self.I_east_Displayed,   self.I_west_Displayed]])
            else : 
                I_bound                  = self.I_Displayed.get_children()[0].get_ydata()
            self.ax_Disk_PhF_I.set_ylim(np.min(I_bound)*(1-marge), np.max(I_bound)*(1+marge))

        if self.PSPF :
            self.PI_Displayed            = self.ax_Disk_PhF_PI.errorbar(self.apSca, self.apSPF/normaP, xerr=self.D_apSca, yerr=np.abs(self.D_apSPF)/normaP, color='black', label='all disk')
            if self.ShowSide:
                self.PI_east_Displayed   = self.ax_Disk_PhF_PI.errorbar(self.epSca, self.epSPF/normeP, xerr=self.D_epSca, yerr=np.abs(self.D_epSPF)/normeP, color='blue',  label='east side', alpha=0.4, ls='dotted')
                self.PI_west_Displayed   = self.ax_Disk_PhF_PI.errorbar(self.wpSca, self.wpSPF/normwP, xerr=self.D_wpSca, yerr=np.abs(self.D_wpSPF)/normwP, color='red',   label='west side', alpha=0.4, ls='dotted')
                PI_bound                 = np.concatenate([line.get_children()[0].get_ydata() for line in [self.PI_Displayed,  self.PI_east_Displayed,  self.PI_west_Displayed]])
            else : 
                PI_bound                 = self.PI_Displayed.get_children()[0].get_ydata()
            self.ax_Disk_PhF_PI.set_ylim(np.min(PI_bound)*(1-marge), np.max(PI_bound)*(1+marge))

        if self.TSPF and self.PSPF:
            self.DoP_Displayed           = self.ax_Disk_DoP.errorbar(self.adSca, self.aDoP, yerr=self.D_aDoP, color='black', label='all disk')
            if self.ShowSide:
                self.DoP_east_Displayed  = self.ax_Disk_DoP.errorbar(self.edSca, self.eDoP, yerr=self.D_eDoP, color='blue', label='east side', alpha=0.4, ls='dotted')
                self.DoP_west_Displayed  = self.ax_Disk_DoP.errorbar(self.wdSca, self.wDoP, yerr=self.D_wDoP, color='red', label='west side', alpha=0.4, ls='dotted')
                DoP_bound                = np.concatenate([line.get_children()[0].get_ydata() for line in [self.DoP_Displayed, self.DoP_east_Displayed, self.DoP_west_Displayed]])
            else : 
                DoP_bound                = self.DoP_Displayed.get_children()[0].get_ydata()
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
        try:
            Fig_img_phF.clear()  # Nettoyer la figure
        except:
            Tool_Could_Also_be_named = 'DRAGyS'
        self.Img_PhF = QWidget()
        self.Img_PhF.setWindowTitle("Phase Functions")
        # Fig_img_phF = Figure()
        Fig_img_phF = plt.figure(figsize = (9, 4))#, dpi = 300)
        ax_img_Extraction   = plt.subplot2grid((1, 5), (0, 0), colspan=2, rowspan=1)
        ax_phF              = plt.subplot2grid((1, 5), (0, 2), colspan=3, rowspan=1)

        ax_phF.clear()
        ax_img_Extraction.clear()
        
        Canvas_Img_PhF = FigureCanvas(Fig_img_phF)
        Toolbar = NavigationToolbar(Canvas_Img_PhF, self)
        layout_Img_PhF = QVBoxLayout()
        folder = f"{self.SPF_Folder}/{self.img_type}_{self.disk_name}.spf"
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


        R_in = float(self.R_in_entry.value())
        R_out = float(self.R_out_entry.value())
        Side = self.Compute_Side()
        pixelscale = float(self.pixelscale_entry.value())
        size = len(self.img_0)
        arcsec_extent = size/2 * pixelscale
        # ax_img_Extraction.imshow(self.img_chose, origin='lower', cmap="gnuplot", norm=colors.SymLogNorm(linthresh=self.thresh_chose), extent=[-arcsec_extent, arcsec_extent, -arcsec_extent, arcsec_extent])
        ax_img_Extraction.imshow(self.img_chose, origin='lower', cmap="gnuplot", norm=colors.LogNorm(), extent=[-arcsec_extent, arcsec_extent, -arcsec_extent, arcsec_extent])

        X_in, Y_in = self.Ellipse(R_in, pixelscale)
        X_out, Y_out = self.Ellipse(R_out, pixelscale)
        ax_img_Extraction.set_facecolor('black')
        ax_img_Extraction.fill(np.append(X_in, X_out[::-1]), np.append(Y_in, Y_out[::-1]), color='gold', alpha=0.4, linestyle='')
        ax_img_Extraction.scatter(X_in,  Y_in,  s=1, c='orange')
        ax_img_Extraction.scatter(X_out, Y_out, s=1, c='orange')

        ax_phF.errorbar(Scatt, PI, xerr=Err_Scatt, yerr=np.abs(Err_PI), marker='.', capsize=2, color='black', label='LB uncorrected', ls='dashed', alpha=0.2)
        ax_phF.errorbar(Scatt, PI_LB,  xerr=Err_Scatt, yerr=np.abs(Err_PI_LB), marker='.', capsize=2, color='black', label='LB corrected')
        if 'MCFOST' in self.file_name:
            MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, Err_MCFOST_I, Err_MCFOST_PI, Err_MCFOST_DoP = Tools.MCFOST_PhaseFunction('/'.join(self.file_path.split('/')[:-2]), self.file_name[:-5], self.normalization)
            NormMCFOSTI  = np.interp(90, MCFOST_Scatt, MCFOST_I)
            NormMCFOSTPI = np.interp(90, MCFOST_Scatt, MCFOST_PI)
            MCFOST_I  = MCFOST_I/NormMCFOSTI
            MCFOST_PI = MCFOST_PI/NormMCFOSTPI
            if self.img_type == 'Total':
                ax_phF.errorbar(MCFOST_Scatt, np.abs(MCFOST_I), yerr=np.abs(Err_MCFOST_I),   ls='dashed', alpha=0.5, color='purple', label='intrinsic')
            else : 
                ax_phF.errorbar(MCFOST_Scatt, np.abs(MCFOST_PI), yerr=np.abs(Err_MCFOST_PI), ls='dashed', alpha=0.5, color='purple', label='intrinsic')

        ax_phF.legend(loc='upper right')
        ax_phF.set_xlabel('Scattering angle [degree]')
        ax_phF.set_ylabel('Normalized Polarized Intensity')
        ax_phF.set_title("Normalized Polarized Phase Function")
        ax_img_Extraction.set_xlabel("$ \Delta $DEC (arcsec)")
        ax_img_Extraction.set_ylabel("$ \Delta $RA (arcsec)")
        ax_img_Extraction.set_title("Polarized Intensity Image")
        layout_Img_PhF.addWidget(Toolbar)
        layout_Img_PhF.addWidget(Canvas_Img_PhF)
        self.Img_PhF.setLayout(layout_Img_PhF)
        Fig_img_phF.subplots_adjust(wspace=0.5)
        Fig_img_phF.tight_layout()
        self.Img_PhF.show()

    def Open_Header(self):
        # options = QFileDialog.Options()
        with fits.open(self.file_path) as hdul:
            header = hdul[0].header
            header_text = repr(header)
            self.h_refeader_window = HeaderWindow(header_text)
            self.h_refeader_window.show()
    
    def LaunchAzimuthRemover(self):
        (x_min, x_max) = self.ax.get_xlim() 
        PixelScale = float(self.pixelscale_entry.value())
        x_min, x_max = x_min/PixelScale + len(self.img_chose)/2, x_max/PixelScale + len(self.img_chose)/2

        AzimuthWindow = AzimuthEllipseApp(self.img_chose, self.thresh_chose, x_min, x_max)
        # AzimuthWindow.show()
        AzimuthWindow.exec()
        self.AzimuthalAngle = np.array(AzimuthWindow.Azimuth.flatten())
        # fig, ax =plt.subplots(1, 1, num='freuiehuiegerhi')
        # ax.scatter(AzimuthWindow.Azimuth, AzimuthWindow.Azimuth)
        # plt.show()

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

class AzimuthEllipseApp(QDialog):
    def __init__(self, image, threshold, x_min, x_max):
        super().__init__()

        self.setWindowTitle('Suppression d\'azimut sur ellipse')
        self.setGeometry(100, 100, 1000, 800)  # Fenêtre plus large pour placer l'historique à droite
        self.Azimuth = np.linspace(0, 359, 360)
        self.azmask  = list(np.argwhere(np.logical_and(self.Azimuth<=0, self.Azimuth>=360)))
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.img_size   = len(image)/2
        self.img_width  = np.abs(x_max - x_min)/2
        self.image      = image
        self.threshold  = threshold
        self.x_min = x_min
        self.x_max = x_max
        # Paramètres de l'ellipse
        self.ellipse_center = (0, 0)
        self.ellipse_width  = self.img_width/2
        self.ellipse_height = self.img_width/2

        # Initialisation de la liste des wedges supprimés
        self.removed_intervals = []
        self.removed_angle = []
        self.current_wedge = None  # Wedge temporaire pendant le clic
        self.is_drawing = False    # Pour savoir si l'utilisateur maintient le clic

        self.draw_ellipse()

        # Layout principal
        self.main_widget = QWidget(self)
        main_layout = QHBoxLayout(self.main_widget)  # Layout horizontal pour placer la figure à gauche et l'historique à droite

        closeAzimuth = QPushButton('Save Azimuth Angle', self)
        closeAzimuth.clicked.connect(self.SaveAzimuth)

        # Ajouter la figure matplotlib
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(closeAzimuth)

        # Zone pour l'historique à droite
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Aligner en haut
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.history_widget)

        # Ajouter la zone de l'historique au layout principal
        main_layout.addWidget(scroll)

        # Connexion des événements de la souris
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Variables pour stocker les points cliqués
        self.x1, self.y1 = None, None
        self.x2, self.y2 = None, None

        self.setLayout(main_layout)  # Définir le layout du QDialog

    def SaveAzimuth(self):
        remaining_angles = self.get_remaining_angles()
        for start, end in remaining_angles:
            self.azmask += list(np.argwhere(np.logical_and(self.Azimuth>=start, self.Azimuth<=end)))
        self.Azimuth = self.Azimuth[self.azmask]
        self.fig.clf()
        self.canvas.flush_events()
        plt.close(self.fig)
        self.accept()

    def draw_ellipse(self):
        """Tracer l'ellipse à chaque mise à jour"""
        self.ax.clear()
        # self.ax.imshow(self.image, origin='lower', cmap="gnuplot", extent=[-self.img_size, self.img_size, -self.img_size, self.img_size], norm=colors.SymLogNorm(linthresh=self.threshold), zorder=-1, alpha=0.5)
        self.ax.imshow(self.image, origin='lower', cmap="gnuplot", extent=[-self.img_size, self.img_size, -self.img_size, self.img_size], norm=colors.LogNorm(), zorder=-1, alpha=0.5)
        self.ax.set_xlim(-self.img_width, self.img_width)
        self.ax.set_ylim(-self.img_width, self.img_width)
        
        # Tracer les angles restants de l'ellipse
        remaining_angles = self.get_remaining_angles()
        for start, end in remaining_angles:
            theta = np.radians(np.arange(start, end+0.1, 1))
            x = self.ellipse_width * np.cos(theta)
            y = self.ellipse_height * np.sin(theta)
            self.ax.plot(x, y, 'b')  # Tracer en bleu les portions restantes de l'ellipse

        # Tracer les wedges supprimés
        for wedge in self.removed_intervals:
            self.ax.add_patch(wedge)

        if self.current_wedge:
            self.ax.add_patch(self.current_wedge)

        # Assurer un affichage carré
        self.ax.set_aspect('equal')
        # self.ax.set_xlim(-6, 6)
        # self.ax.set_ylim(-6, 6)  # carré, même échelle sur les axes
        self.canvas.draw()

    def get_remaining_angles(self):
        """Retourner la liste des intervalles d'angles restants (azimutaux)"""
        full_circle = [(0, 360)]  # L'ellipse entière
        removed_intervals = [(w.theta1, w.theta2) for w in self.removed_intervals]

        # Fonction pour soustraire un intervalle de la liste complète
        def subtract_intervals(intervals, remove):
            result = []
            for start, end in intervals:
                if remove[1] <= start or remove[0] >= end:
                    # Pas de chevauchement
                    result.append((start, end))
                else:
                    # Chevauchement, on découpe l'intervalle
                    if remove[0] > start:
                        result.append((start, remove[0]))
                    if remove[1] < end:
                        result.append((remove[1], end))
            return result

        # Appliquer toutes les suppressions d'angles
        for remove in removed_intervals:
            full_circle = subtract_intervals(full_circle, remove)

        return full_circle

    def on_click(self, event):
        """Lorsque l'utilisateur clique sur la figure"""
        self.x1, self.y1 = event.xdata, event.ydata
        if self.x1 is not None and self.y1 is not None:
            self.is_drawing = True  # Commencer à tracer l'arc

    def on_release(self, event):
        """Lorsque l'utilisateur relâche la souris"""
        if self.is_drawing:
            self.x2, self.y2 = event.xdata, event.ydata
            if self.x1 is not None and self.y1 is not None and self.x2 is not None and self.y2 is not None:
                # Calculer les angles d'azimut correspondants
                angle1 = np.degrees(np.arctan2(self.y1, self.x1))
                angle2 = np.degrees(np.arctan2(self.y2, self.x2))

                # S'assurer que les angles sont dans le bon sens (0-360 degrés)
                if angle1 < 0:
                    angle1 += 360
                if angle2 < 0:
                    angle2 += 360

                # Créer un wedge permanent entre ces deux angles
                wedge = Wedge(self.ellipse_center, self.ellipse_width*1.5, angle1, angle2, color='red', alpha=0.3)
                self.removed_intervals.append(wedge)
                self.add_to_history(angle1, angle2)

            # Réinitialiser le wedge temporaire et redessiner
            self.current_wedge = None
            self.is_drawing = False
            self.draw_ellipse()

    def on_motion(self, event):
        """Lorsque l'utilisateur déplace la souris en maintenant le clic"""
        if self.is_drawing and self.x1 is not None and self.y1 is not None and event.xdata is not None and event.ydata is not None:
            # Calculer l'angle entre le point initial et le point actuel
            angle1 = np.degrees(np.arctan2(self.y1, self.x1))
            angle2 = np.degrees(np.arctan2(event.ydata, event.xdata))

            if angle1 < 0:
                angle1 += 360
            if angle2 < 0:
                angle2 += 360

            # Mettre à jour le wedge temporaire
            self.current_wedge = Wedge(self.ellipse_center, self.ellipse_width*1.5, angle1, angle2, color='red', alpha=0.3)
            self.removed_angle.append(np.where)
            # Redessiner la figure pour montrer le wedge en temps réel
            self.draw_ellipse()

    def add_to_history(self, angle1, angle2):
        """Ajouter une suppression à l'historique avec un bouton d'annulation"""
        hbox = QHBoxLayout()

        label = QLabel(f"Suppression: {round(angle1)}° - {round(angle2)}°")
        btn_remove = QPushButton("X")
        btn_remove.clicked.connect(lambda: self.undo_removal(angle1, angle2, hbox))

        hbox.addWidget(label)
        hbox.addWidget(btn_remove)
        self.history_layout.addLayout(hbox)

    def undo_removal(self, angle1, angle2, hbox):
        """Annuler une suppression d'azimut"""
        # Retirer le wedge correspondant
        self.removed_intervals = [w for w in self.removed_intervals if not (round(w.theta1) == round(angle1) and round(w.theta2) == round(angle2))]
        # Supprimer l'entrée de l'historique
        for i in reversed(range(hbox.count())):
            hbox.itemAt(i).widget().deleteLater()
        self.draw_ellipse()

class Launcher():
    def Run():
        app = QApplication(sys.argv)
        ex = DRAGyS()
        ex.show()
        app.exec()

Launcher.Run()