from PyQt6.QtWidgets import QMessageBox, QTextEdit, QApplication, QScrollArea, QDialog, QSpacerItem, QSizePolicy, QWidget, QSlider, QCheckBox, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QFrame, QProgressBar
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Wedge
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

from matplotlib import colors

from astropy.io import fits
from scipy import ndimage
import sys
import os
import pathlib
import json
import numpy as np
import pickle as pkl
import time

# current_folder = pathlib.Path(__file__).parent
# sys.path.append(current_folder)
try :
    import dragys.tools as Tools
except:
    import tools as Tools
# import tools as Tools

class DRAGyS(QWidget):
    """
    A class to open PyQt6 main GUI of DRAGyS
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the DRAGyS object.
        """
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        Setup all the boolean and PyQt window parameters
        """
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

        # Config Save Folder Buttons
        self.config_file = "config.json"
        self.folderpath = self.get_save_directory()
        if "DRAGyS_Results" not in os.listdir(self.folderpath):
            os.mkdir(f"{self.folderpath}/DRAGyS_Results")
        self.Configlayout = QHBoxLayout()
        self.Configlabel  = QLabel(f"Current directory : \n{self.folderpath}", self)
        self.Configlabel.setEnabled(False)
        self.change_dir_button = QPushButton("Modify")
        self.change_dir_button.clicked.connect(self.change_save_directory)
    

        # Files Browsers Buttons
        self.browse_button = QPushButton('Browse Files', self)
        self.browse_button.clicked.connect(self.browse_files)
        self.file_label = QLabel('No file selected', self)
        self.file_label.setFixedHeight(15)

        # Fitting Buttons
        self.Display_Fit_button = QPushButton('Show Fitting', self)
        self.Display_Fit_button.clicked.connect(self.Show_Fitting)
        self.Display_Fit_button.setEnabled(False)
        self.Compute_Fit_button = QPushButton('Compute Fitting', self)
        self.Compute_Fit_button.clicked.connect(self.Launch_Filtering_Data)
        self.Compute_Fit_button.setEnabled(False)

        # Parameters Box
        self.UseFittingButton       = QPushButton("Polarized Fitting")
        self.Label_Fitting          = QLabel("No fitting file found", self)
        self.UseFittingButton.setEnabled(False)
        self.UseFittingButton.clicked.connect(self.FittingType)
        # self.UseFittingButton.setFixedWidth(180)
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
        self.PowerLawAlphaLine.setDecimals(3)
        self.ErrInclinationLine.setDecimals(2)
        self.ErrPositionAngleLine.setDecimals(2)
        self.ErrAspectRatioLine.setDecimals(3)
        self.ErrHeightLine.setDecimals(2)
        self.ErrRadiusLine.setDecimals(2)
        self.ErrPowerLawAlphaLine.setDecimals(3)
        
        self.InclinationLine.setValue(0)
        self.PositionAngleLine.setValue(0)
        self.AspectRatioLine.setValue(0)
        self.HeightLine.setValue(10)
        self.RadiusLine.setValue(100)
        self.PowerLawAlphaLine.setValue(1.219)      # Based on Avenhaus+2018         and         Kenyon & Hartmann 1987
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

        self.InclinationLine.editingFinished.connect(self.Extraction_Zone)
        self.PositionAngleLine.editingFinished.connect(self.Extraction_Zone)
        self.HeightLine.editingFinished.connect(self.Extraction_Zone)
        self.RadiusLine.editingFinished.connect(self.Extraction_Zone)
        self.PowerLawAlphaLine.editingFinished.connect(self.Extraction_Zone)
        
        # Star position buttons
        self.CheckStar = QCheckBox('Non Centered Star', self)
        self.X_StarPositionLabel = QLabel("x = ", self)
        self.Y_StarPositionLabel = QLabel("y = ", self)
        self.X_StarPosition = QDoubleSpinBox(self)
        self.Y_StarPosition = QDoubleSpinBox(self)
        self.X_StarPosition.setRange(0.0, 10000.0)
        self.Y_StarPosition.setRange(0.0, 10000.0)
        self.X_StarPosition.setFixedWidth(100)
        self.Y_StarPosition.setFixedWidth(100)
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
        self.DataTypeLabel = QLabel("Data Type ?", self)
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
        self.diameter_entry.setValue(8.2)
        self.wavelength_entry.setValue(1.65)
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
        self.R_adjustement = QCheckBox('Display Extraction Zone', self)
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
        self.SPF_Type_button.setEnabled(False)
        self.Compute_PhF_button.setEnabled(False)
        self.SPF_Type_button.setFixedHeight(30)
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


        # Config Directory
        Configbox = QFrame()
        Configbox.setFrameShape(QFrame.Shape.StyledPanel)
        Configbox.setFrameShadow(QFrame.Shadow.Raised)
        configbox = QHBoxLayout()
        configbox.addWidget(self.Configlabel)
        configbox.addWidget(self.change_dir_button)
        Configbox.setLayout(configbox)

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
        FitBoxButton.setAlignment(StarPositionBox, Qt.AlignmentFlag.AlignCenter)
        FitBoxButton.addWidget(self.AzimuthButton)


        VFitParameters = QVBoxLayout()
        FitParameters = QHBoxLayout()
        FitParameters.setAlignment(Qt.AlignmentFlag.AlignRight)
        FitParameters.addLayout(TxtBox)
        FitParameters.addLayout(ValueBox)
        FitParameters.addLayout(PMBox)
        FitParameters.addLayout(Errorbox)
        VFitParameters.addWidget(self.UseFittingButton)
        VFitParameters.addWidget(self.Label_Fitting)
        VFitParameters.addLayout(FitParameters)
        FitParameters.setAlignment(TxtBox,   Qt.AlignmentFlag.AlignRight)
        FitParameters.setAlignment(ValueBox, Qt.AlignmentFlag.AlignRight)
        FitParameters.setAlignment(PMBox,    Qt.AlignmentFlag.AlignRight)
        FitParameters.setAlignment(Errorbox, Qt.AlignmentFlag.AlignRight)

        fitbox0 = QHBoxLayout()
        fitbox0.addLayout(FitBoxButton)
        spacer = QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        fitbox0.addItem(spacer)
        fitbox0.addLayout(VFitParameters)

        fitbox = QVBoxLayout()
        fitbox.addLayout(fitbox0)
        FitBox.setLayout(fitbox)
        VFitParameters.setAlignment(self.Label_Fitting,    Qt.AlignmentFlag.AlignCenter)
    
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
        PhFbox.setAlignment(self.Is_computed, Qt.AlignmentFlag.AlignCenter)

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
        displaybox.addWidget(self.DataTypeLabel)
        displaybox.setAlignment(self.DataTypeLabel, Qt.AlignmentFlag.AlignCenter)
        DisplayBox.setLayout(displaybox)

        # Organisation
        LeftBBox = QFrame()
        LeftBBox.setFixedWidth(500)
        LeftBBox.setFrameShape(QFrame.Shape.StyledPanel)
        LeftBBox.setFrameShadow(QFrame.Shadow.Raised)
        leftbbox = QVBoxLayout()
        leftbbox.addWidget(Configbox)
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
            
    def get_save_directory(self):
        """
        Checks whether a save directory has already been defined.
        If not, asks the user to select one, and saves it.
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
                if "save_directory" in config and os.path.isdir(config["save_directory"]):
                    return config["save_directory"]
        return self.ask_user_for_directory()

    def ask_user_for_directory(self):
        """
        Displays a dialog box to request a directory from the user.
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setWindowTitle("Select a save directory")
        if dialog.exec():
            selected_directory = dialog.selectedFiles()[0]
            self.save_directory_to_config(selected_directory)
            return selected_directory
        else:
            QMessageBox.critical(self, "Error", "No directory selected. Application will close.")
            sys.exit(1)

    def save_directory_to_config(self, directory):
        """
        Saves the selected directory to a config file.
        """
        with open(self.config_file, "w") as f:
            json.dump({"save_directory": directory}, f)
        self.folderpath = directory

    def change_save_directory(self):
        """
        Allows the user to modify the save directory.
        """
        new_directory = self.ask_user_for_directory()
        if "DRAGyS_Results" not in os.listdir(new_directory):
            os.mkdir(f"{new_directory}/DRAGyS_Results")
        self.save_directory_to_config(new_directory)
        self.Configlabel.setText(f"Current directory : {new_directory}")
        QMessageBox.information(self, "Modified directory", f"The directory has been updated : {new_directory}")

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
            distance = Tools.Get_Distance(self.file_path)
            self.distance_entry.setValue(distance)
        except:
            self.distance_entry.setValue(100) 
        
        band = Tools.Get_Band(self.file_path)
        self.wavelength_entry.setValue(band)

        R_in, R_out = Tools.EZ_Radius(self.file_path)
        self.R_in_entry.setValue(R_in)
        self.R_out_entry.setValue(R_out)


        if "SPHERE" in self.file_name.upper():
            self.pixelscale_entry.setValue(0.01225)
        if 'VBB' in self.file_name:
            self.pixelscale_entry.setValue(0.0036)

        self.pixelscale = float(self.pixelscale_entry.value())

    def update_label_fitting(self):
        """
        Update the fitting label for available or not
        """
        if os.path.exists(f'{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit'):
            self.Label_Fitting.setText(f"Fitting file available")
            self.Label_Fitting.setStyleSheet('color: green;')
        else :
            self.Label_Fitting.setText(f"Fitting file not available")
            self.Label_Fitting.setStyleSheet('color: red;')

# ==================================================================
# =====================     File Finder    =========================
# ==================================================================

    def browse_files(self):
        """
        Browse fits file to open data image
        """
        self.R_adjustement.setEnabled(True)
        self.R_adjustement.setChecked(False)
        self.CheckEZ = False
        self.ax.cla()
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
        file_dialog = QFileDialog()
        self.file_path, _ = file_dialog.getOpenFileName(self, 'Select a file', filter='Fichiers FITS (*.fits)')
        self.disk_name = self.file_path.split("/")[-1]
        self.disk_name = self.disk_name[:-5]
        if self.file_path:
            self.Image_Displayed = True
            self.file_name = (self.file_path.split('/'))[-1]
            self.file_label.setText(f"Selected file : {self.file_name}")
            self.Compute_Fit_button.setEnabled(True)
            self.UseFittingButton.setEnabled(True)
            self.SPF_Type_button.setEnabled(True)
            self.Compute_PhF_button.setEnabled(True)
            self.update_label_fitting()
            self.Set_Initial_Values()
            self.Init_image(self.file_path)
            self.SavedParams()
            self.Clickable_Buttons()
        if os.path.exists(f'{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit'):
            self.Display_Fit_button.setEnabled(True)
        else:
            self.Display_Fit_button.setEnabled(False)
        if self.Fitting and self.Fitting.isVisible():
            self.Fitting.close()

# ==================================================================
# =====================  Condition Buttons =========================
# ==================================================================
    def Clickable_Buttons(self):
        """
        Manages the possibility of clicking on buttons.
        """
        if os.path.exists(f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.img_type[0]).lower()}spf"):
            self.Show_disk_PhF_button.setEnabled(True)
            self.Show_img_PhF_button.setEnabled(True)
        else :
            self.Show_disk_PhF_button.setEnabled(False)
            self.Show_img_PhF_button.setEnabled(False)
        
        if os.path.exists(f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.tspf") and os.path.exists(f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.pspf"):
            self.Show_disk_PhF_button.setEnabled(True)

# ==================================================================
# =====================    Image Display   =========================
# ==================================================================
    def Change_View(self, img_show):
        """
        Manages the displayed images.

        Parameters
        ----------
        img_show    :   int
                        image number to display
        """
        self.img_chose    = self.all_img[img_show]
        self.thresh_chose = self.all_thresh[img_show]
        try:
            self.displayed_img.remove()
        except:
            SERGE = "Scattering Extraction from Ring Geometry Estimation"
        if self.Data_Type == "MCFOST_Data":
            self.displayed_img = self.ax.imshow(self.img_chose, extent=[-self.Size, self.Size, -self.Size, self.Size], origin='lower', cmap="inferno", norm=colors.SymLogNorm(linthresh=self.thresh_chose))
        else :
            self.displayed_img = self.ax.imshow(self.img_chose, extent=[-self.Size, self.Size, -self.Size, self.Size], origin='lower', cmap="inferno", norm=colors.LogNorm())
        self.Zoom_Slider_Update(self.ZoomSlider.value())

    def Init_image(self, file_path):
        """
        Initializes images

        Parameters
        ----------
        file_path   :   str
                        full path to the fits file
        """
        [self.img_0,    self.img_1,    self.img_2,    self.img_3,    self.img_4,    self.img_5], [self.thresh_0, self.thresh_1, self.thresh_2, self.thresh_3, self.thresh_4, self.thresh_5], self.Data_Type = Tools.Images_Opener(file_path)
        self.DataTypeLabel.setText((self.Data_Type).replace('_', ' ') + "?")
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
        """
        Controls the zoom on displayed image

        Parameters
        ----------
        value   :   float
                    arbitrary image zoom size
        """
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
        Called when click on "Polarized/Total Fitting", to update displayed parameters
        """
        if self.fit_type   == 'Total':
            self.fit_type  =  'Polarized'
            self.UseFittingButton.setText("Polarized Fitting")
        elif self.fit_type == 'Polarized':
            self.fit_type  =  'Total'
            self.UseFittingButton.setText("Total Fitting")
        if os.path.exists(f'{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit'):
            self.Display_Fit_button.setEnabled(True)
        else:
            self.Display_Fit_button.setEnabled(False)
        self.update_label_fitting()
        self.SavedParams()

    def SavedParams(self):
        """ 
        Called when open a new disk file or when compute fitting is done or when click on "Polarized/Total Fitting", to update displayed parameters
        """
        try :
            with open(f'{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit', 'rb') as fichier:
                Loaded_Data = pkl.load(fichier)
            [  self.incl,   self.r_ref,   self.h_ref,   self.aspect,   self.alpha,   self.PA, _, _, _, _] = Loaded_Data['params']
            [self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, self.D_alpha, self.D_PA, _, _, _, _] = Loaded_Data['Err']
            self.UpdateFitting()
        except :
            self.UpdateParams()

    def UpdateParams(self):
        """
        Called when one parameter is modified
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
        self.aspect   = self.h_ref/self.r_ref
        # self.aspect   = self.h_ref/self.r_ref**self.alpha
        self.D_aspect = self.aspect * np.sqrt((self.D_r_ref/self.r_ref)**2 + (self.D_h_ref/self.h_ref)**2)
        self.UpdateFitting()

    def UpdateFitting(self):
        """
        Called when open a new disk file or when compute fitting is done or when click on "Polarized/Total Fitting", to update displayed parameters 
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
        """
        Controls the star position

        Parameters
        ----------
        state   :   int
                    PyQt Checkbox value (= 2 if checked)
        """
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
        """
        Displays a second PyQt window to show image and fitted ellipse with detected maxima
        """
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

            with open(f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit", 'rb') as fichier:
                Loaded_Data = pkl.load(fichier)
            [  incl,   R,   H,   Aspect,   chi,   PA,   Xe,   Ye,   Xc,   Yc] = Loaded_Data["params"]
            [D_incl, D_R, D_H, D_Aspect, D_Chi, D_PA, D_Xe, D_Ye, D_Xc, D_Yc] = Loaded_Data['Err']
            [f_incl, f_R, f_H, f_Aspect, f_chi, f_PA, f_Xe, f_Ye, f_Xc, f_Yc] = Loaded_Data["first_estim"]
            [Points, All_Points]                      = Loaded_Data["Points"]  

            # X_e, Y_e = Tools.ellipse(incl, PA, H, R, 1.219, R, np.linspace(0, 2*np.pi, 361), x0=center, y0=center)
            
            ax.set_facecolor('black')
            ax.set_title("Numerically Stable Direct \n Least Squares Fitting of Ellipses", fontweight='bold')
            if self.Data_Type == "MCFOST_Data":
                ax.imshow(self.img_chose, origin='lower', extent=[-size, size, -size, size], cmap="inferno", norm=colors.SymLogNorm(linthresh=self.thresh_chose))
            else :
                ax.imshow(self.img_chose, origin='lower', extent=[-size, size, -size, size], cmap="inferno", norm=colors.LogNorm())

            ax.scatter((Points[:,1]-center)*PixelScale, (Points[:,0]-center)*PixelScale,   marker='.',          color='blue') # given points
            # ax.plot( (f_Y_e-center)*PixelScale, (f_X_e-center)*PixelScale, label="first ellipse fit", color='red')
            ax.plot((Ye-center)*PixelScale, (Xe-center)*PixelScale, label="ellipse fit", color='blue')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ax.set_xlabel('X position [pix]')
            ax.set_ylabel('Y position [pix]')
            ax.legend(loc='upper right')
            ax.text(0.01, 0.99, u'i = {:<15} \n PA = {:<15} \n h/r = {:<15}'.format(str(np.round(np.degrees(  incl),3))+' °', str(np.round(np.degrees(  PA),3))+' °', str(np.round(  Aspect, 3))), bbox=dict(boxstyle="round",ec='k',fc='w', alpha=1), color='darkblue', ha='left', va='top',    transform=ax.transAxes)
            ax.text(0.01, 0.01, u'i = {:<15} \n PA = {:<15} \n h/r = {:<15}'.format(str(np.round(np.degrees(f_incl),3))+' °', str(np.round(np.degrees(f_PA),3))+' °', str(np.round(f_Aspect, 3))), bbox=dict(boxstyle="round",ec='k',fc='w', alpha=1), color='darkred',  ha='left', va='bottom', transform=ax.transAxes)
            self.Fitting.show()

    def Launch_Filtering_Data(self):
        """
        Setup and launch the brightness maxima detection on a second PyQt window
        """
        (x_min, x_max) = self.ax.get_xlim() 
        PixelScale = float(self.pixelscale_entry.value())
        x_min, x_max = x_min/PixelScale + len(self.img_0)/2, x_max/PixelScale + len(self.img_0)/2
        D = float(self.diameter_entry.value())
        W = float(self.wavelength_entry.value())
        r_beam = np.degrees((W*1e-6)/D) *3600/PixelScale
        self.Filtering_Window = FilteringWindow(self.disk_name, self.fit_type, self.img_chose, self.thresh_chose, self.Data_Type, x_min, x_max, r_beam, PixelScale, self.folderpath)
        self.Filtering_Window.exec()
        if os.path.exists(f'{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit'):
            self.Display_Fit_button.setEnabled(True)
            self.SavedParams()

# ==================================================================
# ===================    Extraction Zone   =========================
# ==================================================================
    def Ring_Adjust_2(self, state):
        """
        Controls if the extraction zone is displayed or not

        Parameters
        ----------
        state   :   int
                    PyQt Checkbox value (= 2 if checked)
        """
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
        """
        Compute the extraction zone using displayed geometric parameters
        """
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
        """
        Compute ellipses at a given radius using geometric parameters and taking into account the pixelscale

        Parameters
        ----------
        R           :   float
                        Radius in au
        pixelscale  :   float
                        pixelscale in arcsec/pixel
        """
        with open(f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit", 'rb') as fichier:
            Loaded_Data = pkl.load(fichier)
        [_, _, _, _, _, _, _, _, Xc, Yc] = Loaded_Data["params"]
        xs = ys = len(self.img_0)/2
        yc_p, xc_p = Tools.Orthogonal_Prejection((xs, ys), (Yc, Xc), np.pi/2 - self.PA)
        Phi = np.radians(np.linspace(0, 359, 360))
        d = float(self.distance_entry.value())
        FoV_au = d * (648000/np.pi) * np.tan(pixelscale*len(self.img_0)/3600 * np.pi/180)
        pixelscale_au = FoV_au/len(self.img_0)
        R = R / pixelscale_au
        x     = R * np.sin(Phi)
        y     = self.h_ref * (R/self.r_ref)**self.alpha * np.sin(self.incl) - R * np.cos(Phi) * np.cos(self.incl)
        x_rot = (x * np.cos(np.pi - self.PA) - y * np.sin(np.pi - self.PA) + (xc_p-xs)) *pixelscale
        y_rot = (x * np.sin(np.pi - self.PA) + y * np.cos(np.pi - self.PA) + (yc_p-ys)) *pixelscale
        return y_rot, x_rot    
    
    def Compute_Side(self):
        """
        Compute left and right side of the disk with respect to the position angle
        """
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
        """Control how the SPF file will be saved '.tspf' if 'Total' or '.pspf if 'Polarized' 
        """
        if self.SPF_Type_button.text() =='Polarized':
            self.SPF_Type_button.setText('Total')
            self.img_type = 'Total'
        else:
            self.SPF_Type_button.setText("Polarized")
            self.img_type = "Polarized"
        self.Clickable_Buttons()
        if self.Fitting and self.Fitting.isVisible():
            self.Fitting.close()

    def Run_SPF(self):
        """
        Setup and launch the SPF computation on the defined extraction zone and spf file '.tspf' or '.pspf'
        """
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
            with open(f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.fit_type[0]).lower()}fit", 'rb') as fichier:
                Loaded_Data = pkl.load(fichier)
            [_, _, _, _, _, _, _, _, Xc, Yc] = Loaded_Data["params"]
        else:
            Xc, Yc = None, None
        start = time.time()
        self.total_steps = (2 * 300*360 + 3*len(self.img_0)*len(self.img_0) + 2*n_bin) * 7
        self.computation_step = 0

        Corono_mask = Tools.Correct_Corono_Transmission(self.img_chose, W, pixelscale, output="R2_mask")

        list_params_SPF = [ [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA,             self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask, "Total"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl - self.D_incl, self.PA,             self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask, "Incl"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl + self.D_incl, self.PA,             self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask, "Incl"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA - self.D_PA, self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask, "PA"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA + self.D_PA, self.r_ref, self.h_ref, self.aspect,                    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask, "PA"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA,             self.r_ref, self.h_ref, self.aspect - self.D_aspect,    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask, "Aspect"],
                            [self.img_chose, distance, pixelscale, r_beam, (xs, ys), (Xc, Yc), self.incl,               self.PA,             self.r_ref, self.h_ref, self.aspect + self.D_aspect,    self.alpha, self.D_incl, self.D_r_ref, self.D_h_ref, self.D_aspect, R_in, R_out, n_bin, self.AzimuthalAngle, Corono_mask, "Aspect"]]
        Tools.Compute_SPF(list_params_SPF, self.folderpath, self.file_name, self.img_type)
        end = time.time()
        self.Is_computed.setText(" SPF computed - time = " +str(np.round(end-start, 2)) + ' seconds')
        self.Is_computed.setStyleSheet('color: green')
        self.Show_disk_PhF_button.setEnabled(True)
        self.Show_img_PhF_button.setEnabled(True)

# ==================================================================
# ===================    Display Results   =========================
# ==================================================================
    def Show_disk_PhaseFunction(self):
        """
        Displays SPF in Total and/or Polarized intensity with Degree of Polarization if both are computed.
        """
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
        """
        Controls if the SPF are normalized

        Parameters
        ----------
        state   :   int
                    PyQt Checkbox value (= 2 if checked)
        """
        if state == 2:
            self.NormType = '90'
            self.ax_Disk_PhF_I.autoscale()
        else:
            self.NormType = 'None'
            self.ax_Disk_PhF_I.autoscale()
        self.Update_SPF()
    
    def LBCheck(self, state):
        """
        Controls if the Limb Brightenong effect is corrected or not

        Parameters
        ----------
        state   :   int
                    PyQt Checkbox value (= 2 if checked)
        """
        if state == 2:
            self.LBCorrected = True
        else:
            self.LBCorrected = False
        self.Update_SPF()
        
    def SideCheck(self, state):
        """
        Controls if SPF for each sides are displayed or not

        Parameters
        ----------
        state   :   int
                    PyQt Checkbox value (= 2 if checked)
        """
        if state == 2:
            self.ShowSide = True
        else:
            self.ShowSide = False
        self.Update_SPF()
    
    def LogCheck(self, state):
        """
        Controls if the SPF are in log scale

        Parameters
        ----------
        state   :   int
                    PyQt Checkbox value (= 2 if checked)
        """
        if state == 2:
            self.ax_Disk_PhF_I.set_yscale('log')
            self.ax_Disk_PhF_PI.set_yscale('log')
        else:
            self.ax_Disk_PhF_I.set_yscale('linear')
            self.ax_Disk_PhF_PI.set_yscale('linear')
        self.Update_SPF()

    def MCFOSTCheck(self, state):
        """
        Controls if the MCFOST SPF is displayed or not

        Parameters
        ----------
        state   :   int
                    PyQt Checkbox value (= 2 if checked)
        """
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
        """
        Update SPF displayed with respect to all checkbox (Normalize, Limb Brightening, Side, Log, MCFOST)
        """
        self.ax_Disk_PhF_I.cla()
        self.ax_Disk_PhF_PI.cla()
        self.ax_Disk_DoP.cla()
        self.ax_Disk_DoP.axvline(90, c='k', ls='dotted', alpha=0.3)
        self.ax_Disk_DoP.set_xlim(10, 170)
        self.ax_Disk_PhF_I.set_ylabel("Total Flux")
        self.ax_Disk_PhF_I.set_xlabel("Scattering Angle (deg)")
        self.ax_Disk_PhF_PI.set_ylabel("Polarized Flux")
        self.ax_Disk_PhF_PI.set_xlabel("Scattering Angle (deg)")
        self.ax_Disk_DoP.set_ylabel("Degree of Polarization")
        self.ax_Disk_DoP.set_xlabel("Scattering Angle (deg)")

        total_folder     = f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.tspf"
        polarized_folder = f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.pspf"
        try :
            self.atSca, self.atSPF, self.atLB, self.D_atSca, self.D_atSPF, self.D_atLB = Tools.Get_SPF(total_folder, side='All',  LBCorrected=self.LBCorrected, norm=self.NormType)
            self.etSca, self.etSPF, self.etLB, self.D_etSca, self.D_etSPF, self.D_etLB = Tools.Get_SPF(total_folder, side='East', LBCorrected=self.LBCorrected, norm=self.NormType)
            self.wtSca, self.wtSPF, self.wtLB, self.D_wtSca, self.D_wtSPF, self.D_wtLB = Tools.Get_SPF(total_folder, side='West', LBCorrected=self.LBCorrected, norm=self.NormType)
            self.TSPF = True
        except:
            self.TSPF = False
        try:
            self.apSca, self.apSPF, self.apLB, self.D_apSca, self.D_apSPF, self.D_apLB = Tools.Get_SPF(polarized_folder, side='All',  LBCorrected=self.LBCorrected, norm=self.NormType)
            self.epSca, self.epSPF, self.epLB, self.D_epSca, self.D_epSPF, self.D_epLB = Tools.Get_SPF(polarized_folder, side='East', LBCorrected=self.LBCorrected, norm=self.NormType)
            self.wpSca, self.wpSPF, self.wpLB, self.D_wpSca, self.D_wpSPF, self.D_wpLB = Tools.Get_SPF(polarized_folder, side='West', LBCorrected=self.LBCorrected, norm=self.NormType)
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
            MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, MCFOST_Err_I, MCFOST_Err_PI, MCFOST_Err_DoP = Tools.MCFOST_PhaseFunction('/'.join(self.file_path.split('/')[:-2]), True)
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

    def Show_img_PhF(self):
        """
        Displays normalized SPF, corrected and uncorrected from Limb Brightening, with fits image and extraction zone
        """
        try:
            Fig_img_phF.clear()  # Nettoyer la figure
        except:
            Tool_Could_Also_be_named = 'DRAGyS'
        self.Img_PhF = QWidget()
        self.Img_PhF.setWindowTitle("Phase Functions")
        Fig_img_phF = plt.figure(figsize = (9, 4))#, dpi = 300)
        ax_img_Extraction   = plt.subplot2grid((1, 5), (0, 0), colspan=2, rowspan=1)
        ax_phF              = plt.subplot2grid((1, 5), (0, 2), colspan=3, rowspan=1)

        ax_phF.clear()
        ax_img_Extraction.clear()
        
        Canvas_Img_PhF = FigureCanvas(Fig_img_phF)
        Toolbar = NavigationToolbar(Canvas_Img_PhF, self)
        layout_Img_PhF = QVBoxLayout()
        folder = f"{self.folderpath}/DRAGyS_Results/{self.disk_name}.{(self.img_type[0]).lower()}spf"
        Scatt,      PI,      LB,      Err_Scatt,      Err_PI,      Err_LB      = Tools.Get_SPF(folder, side='All')
        Scatt_east, PI_east, LB_east, Err_Scatt_east, Err_PI_east, Err_LB_east = Tools.Get_SPF(folder, side='East')
        Scatt_west, PI_west, LB_west, Err_Scatt_west, Err_PI_west, Err_LB_west = Tools.Get_SPF(folder, side='West')
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
        if self.Data_Type == 'MCFOST_Data':
            ax_img_Extraction.imshow(self.img_chose, origin='lower', cmap="inferno", norm=colors.SymLogNorm(linthresh=self.thresh_chose), extent=[-arcsec_extent, arcsec_extent, -arcsec_extent, arcsec_extent])
        else :
            ax_img_Extraction.imshow(self.img_chose, origin='lower', cmap="inferno", norm=colors.LogNorm(), extent=[-arcsec_extent, arcsec_extent, -arcsec_extent, arcsec_extent])

        X_in, Y_in = self.Ellipse(R_in, pixelscale)
        X_out, Y_out = self.Ellipse(R_out, pixelscale)
        ax_img_Extraction.set_facecolor('black')
        ax_img_Extraction.fill(np.append(X_in, X_out[::-1]), np.append(Y_in, Y_out[::-1]), color='gold', alpha=0.4, linestyle='')
        ax_img_Extraction.scatter(X_in,  Y_in,  s=1, c='orange')
        ax_img_Extraction.scatter(X_out, Y_out, s=1, c='orange')

        ax_phF.errorbar(Scatt, PI, xerr=Err_Scatt, yerr=np.abs(Err_PI), marker='.', capsize=2, color='black', label='LB uncorrected', ls='dashed', alpha=0.2)
        ax_phF.errorbar(Scatt, PI_LB,  xerr=Err_Scatt, yerr=np.abs(Err_PI_LB), marker='.', capsize=2, color='black', label='LB corrected')
        if 'MCFOST' in self.file_name:
            MCFOST_Scatt, MCFOST_I, MCFOST_PI, MCFOST_DoP, Err_MCFOST_I, Err_MCFOST_PI, Err_MCFOST_DoP = Tools.MCFOST_PhaseFunction('/'.join(self.file_path.split('/')[:-2]), self.normalization)
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
        (x_min, x_max) = self.ax.get_xlim()
        ax_img_Extraction.set_xlim(x_min, x_max)
        ax_img_Extraction.set_ylim(x_min, x_max)
        layout_Img_PhF.addWidget(Toolbar)
        layout_Img_PhF.addWidget(Canvas_Img_PhF)
        self.Img_PhF.setLayout(layout_Img_PhF)
        Fig_img_phF.subplots_adjust(wspace=0.5)
        Fig_img_phF.tight_layout()
        self.Img_PhF.show()

    def Open_Header(self):
        """
        Displays Header of the fits file
        """
        with fits.open(self.file_path) as hdul:
            header = hdul[0].header
            header_text = repr(header)
            self.h_refeader_window = HeaderWindow(header_text)
            self.h_refeader_window.show()
    
    def LaunchAzimuthRemover(self):
        """
        Launch PyQt window for removing some angle where you don not want to extract the SPF
        """
        (x_min, x_max) = self.ax.get_xlim() 
        PixelScale = float(self.pixelscale_entry.value())
        x_min, x_max = x_min/PixelScale + len(self.img_chose)/2, x_max/PixelScale + len(self.img_chose)/2
        AzimuthWindow = AzimuthEllipseApp(self.img_chose, self.thresh_chose, self.Data_Type, x_min, x_max)
        AzimuthWindow.exec()
        self.AzimuthalAngle = np.array(AzimuthWindow.Azimuth.flatten())

class HeaderWindow(QWidget):
    """
    A class to open secondary PyQt window to display fits file header
    """
    def __init__(self, header_text):
        """
        Open fits file header
        """
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
    """
    A class to open a secondary PyQt window to remove Azimuth angle from SPF computation

    Parameters
    ----------

    image           :   numpy.ndarray
                        fits image
    threshold       :   float
                        threshold for SymLogNorm in imshow function to avoid vmin > vmax if LogNorm...
    Data_Type       :   str
                        change imshow norm to colors.SymLogNorm() if Data_Type=="MCFOST_Data"
    x_min, x_max    :   float
                        define the image size with respect to zoom value defined in the main PyQt window
    """
    def __init__(self, image, threshold, Data_Type, x_min, x_max):
        """
        initializes the secondary window
        """
        super().__init__()

        self.setWindowTitle('Suppression d\'azimut sur ellipse')
        self.setGeometry(100, 100, 1000, 800)
        self.Azimuth = np.linspace(0, 359, 360)
        self.azmask  = list(np.argwhere(np.logical_and(self.Azimuth<=0, self.Azimuth>=360)))
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.img_size   = len(image)/2
        self.img_width  = np.abs(x_max - x_min)/2
        self.image      = image
        self.threshold  = threshold
        self.Data_Type  = Data_Type
        self.x_min = x_min
        self.x_max = x_max

        self.ellipse_center = (0, 0)
        self.ellipse_width  = self.img_width/2
        self.ellipse_height = self.img_width/2

        self.removed_intervals = []
        self.removed_angle = []
        self.current_wedge = None
        self.is_drawing = False

        self.draw_ellipse()

        self.main_widget = QWidget(self)
        main_layout = QHBoxLayout(self.main_widget)

        closeAzimuth = QPushButton('Save Azimuth Angle', self)
        closeAzimuth.clicked.connect(self.SaveAzimuth)

        main_layout.addWidget(self.canvas)
        main_layout.addWidget(closeAzimuth)

        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.history_widget)

        main_layout.addWidget(scroll)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.x1, self.y1 = None, None
        self.x2, self.y2 = None, None

        self.setLayout(main_layout)

    def SaveAzimuth(self):
        """
        save azimuth angle to use during the SPF computation
        """
        remaining_angles = self.get_remaining_angles()
        for start, end in remaining_angles:
            self.azmask += list(np.argwhere(np.logical_and(self.Azimuth>=start, self.Azimuth<=end)))
        self.Azimuth = self.Azimuth[self.azmask]
        self.fig.clf()
        self.canvas.flush_events()
        plt.close(self.fig)
        self.accept()

    def draw_ellipse(self):
        """
        Plots ellipse at each update
        """
        self.ax.clear()
        if self.Data_Type == "MCFOST_Data":
            self.ax.imshow(self.image, origin='lower', cmap="inferno", extent=[-self.img_size, self.img_size, -self.img_size, self.img_size], norm=colors.SymLogNorm(linthresh=self.threshold), zorder=-1, alpha=0.5)
        else :
            self.ax.imshow(self.image, origin='lower', cmap="inferno", extent=[-self.img_size, self.img_size, -self.img_size, self.img_size], norm=colors.LogNorm(), zorder=-1, alpha=0.5)
        self.ax.set_xlim(-self.img_width, self.img_width)
        self.ax.set_ylim(-self.img_width, self.img_width)
        
        remaining_angles = self.get_remaining_angles()
        for start, end in remaining_angles:
            theta = np.radians(np.arange(start, end+0.1, 1))
            x = self.ellipse_width * np.cos(theta)
            y = self.ellipse_height * np.sin(theta)
            self.ax.plot(x, y, 'b')

        for wedge in self.removed_intervals:
            self.ax.add_patch(wedge)

        if self.current_wedge:
            self.ax.add_patch(self.current_wedge)

        self.ax.set_aspect('equal')
        self.canvas.draw()

    def get_remaining_angles(self):
        """
        Return list of remaining azimuthal angle intervals

        Returns
        -------

        list
            list of remaining azimuthal angle intervals
        """
        full_circle = [(0, 360)]
        removed_intervals = [(w.theta1, w.theta2) for w in self.removed_intervals]

        def subtract_intervals(intervals, remove):
            result = []
            for start, end in intervals:
                if remove[1] <= start or remove[0] >= end:
                    result.append((start, end))
                else:
                    if remove[0] > start:
                        result.append((start, remove[0]))
                    if remove[1] < end:
                        result.append((remove[1], end))
            return result

        for remove in removed_intervals:
            full_circle = subtract_intervals(full_circle, remove)

        return full_circle

    def on_click(self, event):
        """
        Called when click on figure
        """
        self.x1, self.y1 = event.xdata, event.ydata
        if self.x1 is not None and self.y1 is not None:
            self.is_drawing = True

    def on_release(self, event):
        """
        Called when click release
        """
        if self.is_drawing:
            self.x2, self.y2 = event.xdata, event.ydata
            if self.x1 is not None and self.y1 is not None and self.x2 is not None and self.y2 is not None:
                angle1 = np.degrees(np.arctan2(self.y1, self.x1))
                angle2 = np.degrees(np.arctan2(self.y2, self.x2))
                if angle1 < 0:
                    angle1 += 360
                if angle2 < 0:
                    angle2 += 360

                wedge = Wedge(self.ellipse_center, self.ellipse_width*1.5, angle1, angle2, color='red', alpha=0.3)
                self.removed_intervals.append(wedge)
                self.add_to_history(angle1, angle2)

            self.current_wedge = None
            self.is_drawing = False
            self.draw_ellipse()

    def on_motion(self, event):
        """
        Called hen the user moves the mouse while holding down the click
        """
        if self.is_drawing and self.x1 is not None and self.y1 is not None and event.xdata is not None and event.ydata is not None:
            angle1 = np.degrees(np.arctan2(self.y1, self.x1))
            angle2 = np.degrees(np.arctan2(event.ydata, event.xdata))

            if angle1 < 0:
                angle1 += 360
            if angle2 < 0:
                angle2 += 360

            self.current_wedge = Wedge(self.ellipse_center, self.ellipse_width*1.5, angle1, angle2, color='red', alpha=0.3)
            self.removed_angle.append(np.where)
            self.draw_ellipse()

    def add_to_history(self, angle1, angle2):
        """
        Add a deletion to the history with an undo button

        Parameters
        ----------

        angle1, angle2      :   float
                                azimuth supression interval limits 
        """
        hbox = QHBoxLayout()

        label = QLabel(f"Suppression: {round(angle1)}° - {round(angle2)}°")
        btn_remove = QPushButton("X")
        btn_remove.clicked.connect(lambda: self.undo_removal(angle1, angle2, hbox))

        hbox.addWidget(label)
        hbox.addWidget(btn_remove)
        self.history_layout.addLayout(hbox)

    def undo_removal(self, angle1, angle2, hbox):
        """
        Cancel azimuth deletion

        Parameters
        ----------

        angle1, angle2      :   float
                                azimuth supression interval limits 
        hbox                :   PyQt.QHBoxLayout
                                Layout where azimuth angle range are displayed
        """
        self.removed_intervals = [w for w in self.removed_intervals if not (round(w.theta1) == round(angle1) and round(w.theta2) == round(angle2))]
        for i in reversed(range(hbox.count())):
            hbox.itemAt(i).widget().deleteLater()
        self.draw_ellipse()

class FilteringWindow(QDialog):
    """
    A class to detect maxima and allow filtering of some spurious points
    
    Parameters
    ----------

    disk_name       :   str
                        fits file name without extension

    img_name        :   str
                        Type of data "Polarized" or "Total", to save fitting data in '.pfit' or '.tfit' respectively

    image           :   numpy.ndarray
                        fits image

    threshold       :   float
                        threshold for SymLogNorm in imshow function to avoid vmin > vmax if LogNorm...

    Data_Type       :   str
                        change imshow norm to colors.SymLogNorm() if Data_Type=="MCFOST_Data"

    x_min, x_max    :   float
                        define the image size with respect to zoom value defined in the main PyQt window

    r_beam          :   float
                        radius of the beam size ~lambda/D with lambda the wavelength and D the telescope diameter. Used to homogenize peak detection on the ellipse

    folderpath      :   str
                        full path to the saving folder (where all fitting and spf data are saved) displayed on top left of the main window

    """

    def __init__(self, disk_name, img_name, image, threshold, Data_Type, x_min, x_max, r_beam, pixelscale, folderpath, parent=None):
        super(FilteringWindow, self).__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
        self.resize(1000, 600)
        self.nb_steps = 100
        self.disk_name = disk_name
        band = Tools.Get_Band(disk_name)
        self.image     = Tools.Correct_Corono_Transmission(image, band, pixelscale, output='img')
        self.threshold = threshold
        self.Data_Type = Data_Type
        self.img_name  = img_name
        self.Lim_Radius = int(len(image)/2 - 1)
        self.x_min, self.x_max = x_min, x_max
        self.R_max = int(x_max - len(self.image)/2)
        self.r_beam = r_beam

        self.folderpath = folderpath
        self.initUI()

    def initUI(self):
        """
        Setup the secondary PyQt window for filtering peak detection
        """
        self.setWindowTitle('Filtering Pixel position Data Window')
        try:
            self.Filtering_Fig.clear()  # Nettoyer la figure
        except:
            Tool_Could_Also_be_named = 'DRAGyS'
        self.Filtering_Fig = Figure()
        self.Filtering_ax  = self.Filtering_Fig.add_subplot(111)
        self.Filtering_Canvas = FigureCanvas(self.Filtering_Fig)
        self.Filtering_Canvas.setParent(self)
        
        close_button = QPushButton('Save Data - Close', self)
        close_button.clicked.connect(self.Continue)
        close_button.setFixedHeight(40)
        close_button.setFixedWidth(300)
        
        self.Filtering_Canvas.mpl_connect('button_press_event', self.on_press)
        self.Filtering_Canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.Filtering_Canvas.mpl_connect('button_release_event', self.on_release)

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
        self.Prominence_slider.setValue(-100)
        self.Prominence_value = 10**(-100/100)
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
        self.Width_slider.setValue(5000)
        self.Width_value = 5
        self.Width_value_Label = QLabel(str(self.Width_value))
        self.Width_slider.setSingleStep(1)
        self.Width_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.Width_Label.setFixedWidth(200)
        self.Width_value_Label.setFixedWidth(100)
        self.Width_slider.setFixedWidth(300)

        self.HighPass_Label  = QLabel("High Pass Parameter : ", self)
        self.HighPass_slider = QSlider(Qt.Orientation.Horizontal)
        self.HighPass_slider.setMinimum(0)
        self.HighPass_slider.setMaximum(1000)
        self.HighPass_slider.setValue(0)
        self.HighPass_value = 1
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
        self.MaxCutRad_slider.setValue(self.Lim_Radius)
        self.MaxCutRad_value = self.Lim_Radius
        self.MaxCutRad_value_Label = QLabel(str(self.MaxCutRad_value))
        self.MaxCutRad_slider.setSingleStep(1)
        self.MaxCutRad_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.MaxCutRad_Label.setFixedWidth(200)
        self.MaxCutRad_value_Label.setFixedWidth(100)
        self.MaxCutRad_slider.setFixedWidth(300)

        self.NbAzimuth_Label  = QLabel("nb azimuth : ", self)
        self.NbAzimuth_slider = QSlider(Qt.Orientation.Horizontal)
        self.NbAzimuth_slider.setMinimum(10)
        self.NbAzimuth_slider.setMaximum(360)
        self.NbAzimuth_slider.setValue(90)
        self.NbAzimuth_value = 90
        self.NbAzimuth_value_Label = QLabel(str(self.NbAzimuth_value))
        self.NbAzimuth_slider.setSingleStep(10)
        self.NbAzimuth_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self.NbAzimuth_Label.setFixedWidth(200)
        self.NbAzimuth_value_Label.setFixedWidth(100)
        self.NbAzimuth_slider.setFixedWidth(300)


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
        self.NbAzimuth_slider.valueChanged.connect(lambda  value : self.Change_Fit(value, Type='NbAzimuth'))
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
        self.NbAzimuth_Label.setFixedHeight(10)
        self.NbAzimuth_value_Label.setFixedHeight(10)

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

        NbAzimuthLayout = QHBoxLayout()
        NbAzimuthLayout.addWidget(self.NbAzimuth_Label)
        NbAzimuthLayout.addWidget(self.NbAzimuth_value_Label)
        Filtering_Layout.addLayout(NbAzimuthLayout)
        Filtering_Layout.addWidget(self.NbAzimuth_slider)

        Filtering_Layout.addLayout(MethodButton)
        Filtering_Layout.addWidget(close_button)
        Filtering_Layout.addWidget(self.progress)

        Figure_Layout = QVBoxLayout()

        Figure_Layout.addWidget(self.Filtering_Canvas)

        All_Layout = QHBoxLayout()
        All_Layout.addLayout(Filtering_Layout)
        All_Layout.addLayout(Figure_Layout)
        self.setLayout(All_Layout)

        self.rect_start = None
        self.rect_preview = None
        X, Y = Tools.Max_pixel(self.image, R_max=self.R_max, gaussian_filter=self.gaussian_value, smooth_filter=self.smooth_value)
        self.is_drawing  = False
        self.line        = None
        self.points      = []
        self.point_All   = np.column_stack((Y, X))
        self.point_cloud = np.column_stack((Y, X))
        
        self.Filtering_ax.set_facecolor('k')
        if self.Data_Type == "MCFOST_Data":
            self.displayed_image = self.Filtering_ax.imshow(self.image, cmap='inferno', norm=colors.SymLogNorm(linthresh=self.threshold))
        else :
            self.displayed_image = self.Filtering_ax.imshow(self.image, cmap='inferno', norm=colors.LogNorm())
        self.Filtering_ax.set_xlim(self.x_min, self.x_max)
        self.Filtering_ax.set_ylim(self.x_min, self.x_max)
        self.scatter = self.Filtering_ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], edgecolor='k', color='cyan', s=5)
        self.Change_Fit(1, 'None')

    def Change_Fit(self, value, Type):
        """
        Update the peak detected with respect to filters value

        Parameters
        ----------

        value       :   int
                        value from slider
        Type        :   str
                        define the type of filter whose value is to be changed
        """
        if Type=='Gaussian':
            self.gaussian_value = value /100
            self.Gaussian_value_Label.setText(str(self.gaussian_value))
            self.displayed_image.remove()
            if self.Data_Type == "MCFOST_Data":
                self.displayed_image = self.Filtering_ax.imshow(ndimage.gaussian_filter(self.image , sigma = self.gaussian_value), cmap='inferno', norm=colors.SymLogNorm(linthresh=self.threshold))
            else :
                self.displayed_image = self.Filtering_ax.imshow(ndimage.gaussian_filter(self.image , sigma = self.gaussian_value), cmap='inferno', norm=colors.LogNorm())

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
        elif Type=='NbAzimuth':
            self.NbAzimuth_value = value
            self.NbAzimuth_value_Label.setText(str(self.NbAzimuth_value))
        elif Type=="Azimuthal" or Type=='Diagonal' or Type=='Antidiagonal' or Type=='Vertical' or Type=='Horizontal':
            self.Method_Value = Type
        self.scatter.remove()
        X, Y = Tools.Max_pixel(self.image, R_max = self.R_max,  gaussian_filter = self.gaussian_value, 
                                                                smooth_filter   = self.smooth_value, 
                                                                prominence      = self.Prominence_value, 
                                                                distance        = self.Distance_value, 
                                                                width           = self.Width_value, 
                                                                threshold       = None,
                                                                HighPass        = self.HighPass_value,
                                                                Mincut_Radius   = self.MinCutRad_value,
                                                                Maxcut_Radius   = self.MaxCutRad_value,
                                                                nb_phi          = self.NbAzimuth_value,
                                                                method          = self.Method_Value)
        self.point_All   = np.column_stack((Y, X))
        self.point_cloud = np.column_stack((Y, X))
        self.scatter = self.Filtering_ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], edgecolor='k', color='cyan', s=5)
        self.Filtering_Canvas.draw()

    def on_press(self, event):
        """
        Start a new trace when the mouse button is pressed down
        """
        self.is_drawing = True
        self.points = [(event.xdata, event.ydata)]  # Initialiser avec le premier point

    def on_release(self, event):
        """
        End trace when mouse button is released
        """
        self.is_drawing = False

        if len(self.points) > 2:  # S'assurer qu'il y a au moins 3 points pour former une zone
            # Ajouter le point de départ pour fermer la forme si nécessaire
            if self.points[0] != self.points[-1]:
                self.points.append(self.points[0])  # Fermer la forme

            # Créer et ajouter le polygone temporairement pour détection
            polygon = Polygon(self.points, closed=True, color='red', alpha=0.3)
            self.Filtering_ax.add_patch(polygon)
            self.Filtering_Canvas.draw()

            # Supprimer les points à l'intérieur de la forme
            self.remove_points_in_polygon(self.points)

            # Supprimer le polygone de la figure
            polygon.remove()

            # Supprimer également la ligne noire du tracé
            if self.line:
                self.line.remove()
                self.line = None

        # Réinitialiser pour un nouveau tracé et mettre à jour l'affichage
        self.points = []
        self.Filtering_Canvas.draw()

    def on_motion(self, event):
        """
        Continuous drawing as the mouse moves
        """
        if self.is_drawing and event.xdata is not None and event.ydata is not None:
            # Ajouter le point actuel à la liste
            self.points.append((event.xdata, event.ydata))

            # Afficher le tracé temporaire pour voir la forme en cours
            if self.line:
                self.line.remove()  # Supprimer le tracé précédent
            self.line, = self.Filtering_ax.plot([p[0] for p in self.points], [p[1] for p in self.points], color='papayawhip')
            self.Filtering_Canvas.draw()

    def remove_points_in_polygon(self, polygon_points):
        """
        Delete the points inside the drawn polygon.

        Parameters
        ----------

        polygon_points      :   list
                                list of points on which to delete maxima
        """
        # Convert polygon points into a Path object for verification
        path = Path(polygon_points)
        
        # Determine which points of the cloud are in the polygon
        inside = path.contains_points(self.point_cloud)
        
        # Keep only points outside the polygon
        self.point_cloud = self.point_cloud[~inside]
        
        # Update point cloud display
        self.scatter.set_offsets(self.point_cloud)

    def Continue(self):
        """
        Finish the peak fitting by estimating the geometric parameters. All points, selected points, ellipse estimates, estimated geometric parameters, their errors, the image, and filter parameters are saved in a “{TargetName}.{Type}fit” file.
        """
        Fit_Name     = f'{self.disk_name}.{(self.img_name[0]).lower()}fit'
        x0 = y0      = int(len(self.image)/2)

        nb_rand     = 100
        nb_peaks    = len(self.point_cloud)
        peaks_range = np.arange(0, nb_peaks, 4)
        nb_peaks_range = len(peaks_range)
        nb_points   = 100

        step = 0

        Inclination   = []
        PositionAngle = []
        Radius        = []
        Height        = []
        Aspect        = []

        X_center  = []
        Y_center  = []
        X_ellipse = []
        Y_ellipse = []

        f_incl, f_PA, f_R, f_H, f_H_R, f_Xe, f_Ye, f_Xc, f_Yc = Tools.Ellipse_Estimation(self.point_cloud, x0, y0)

        for dec in peaks_range:   # Offset starting point from 4 index position, to remove too close points 
            filtered_Points = Tools.filtrer_nuage_points(self.point_cloud, self.r_beam, dec=dec)
            incl, PA, R, H, aspect, Xe, Ye, Xc, Yc = Tools.Ellipse_Estimation(filtered_Points, x0, y0)
            std_ellipse = Tools.Std_Ellipse(filtered_Points, Xe, Ye, Xc, Yc)
            for n in range(nb_rand):
                Random_Radius       = np.random.normal(R, std_ellipse, nb_points)
                Random_Phi          = np.random.uniform(0, 2*np.pi, nb_points)
                Xrand, Yrand = Tools.ellipse(incl, PA, H, R, 1.219, Random_Radius, Random_Phi, x0=x0, y0=y0)
                inclination, positionangle, radius, height, h_r, Xe, Ye, Xc, Yc = Tools.Ellipse_Estimation(np.column_stack((Xrand, Yrand)), x0, y0)
                Inclination.append(inclination)
                PositionAngle.append(positionangle)
                Radius.append(radius)
                Height.append(height)
                Aspect.append(h_r)
                X_center.append(Xc)
                Y_center.append(Yc)
                X_ellipse.append(Xe)
                Y_ellipse.append(Ye)
                step += 1
                self.progress.setValue(int(100*step/(nb_rand*nb_peaks_range)))

        incl = np.mean(Inclination)
        PA   = np.mean(PositionAngle)
        R    = np.mean(Radius)
        H    = np.mean(Height)
        H_R  = np.mean(Aspect)

        D_incl = np.std(Inclination)
        D_PA   = np.std(PositionAngle)
        D_R    = np.std(Radius)
        D_H    = np.std(Height)
        D_H_R  = np.std(Aspect)

        Xc = np.mean(X_center)
        Yc = np.mean(Y_center)

        D_Xc = np.std(X_center)
        D_Yc = np.std(Y_center)

        Xe = np.mean(np.array(X_ellipse), axis=0)
        Ye = np.mean(np.array(Y_ellipse), axis=0)

        D_Xe = np.std(np.array(X_ellipse), axis=0)
        D_Ye = np.std(np.array(Y_ellipse), axis=0)

        Data_to_Save    = { "params"        : [  incl,   R,   H,   H_R, 1.219,   PA,   Xe,   Ye,   Xc,   Yc],      # Based on Avenhaus et al. 2018         and         Kenyon & Hartmann 1987 
                            "Err"           : [D_incl, D_R, D_H, D_H_R, 0.000, D_PA, D_Xe, D_Ye, D_Xc, D_Yc], 
                            "first_estim"   : [f_incl, f_R, f_H, f_H_R, 1.219, f_PA, f_Xe, f_Ye, f_Xc, f_Yc],
                            'Points'        : [self.point_cloud, self.point_All], 
                            "Data"          : self.image,
                            "Filters"       : {"gaussian_filter" : self.gaussian_value, 
                                                "smooth_filter"   : self.smooth_value, 
                                                "prominence"      : self.Prominence_value,
                                                "distance"        : self.Distance_value, 
                                                "width"           : self.Width_value, 
                                                "HighPass"        : self.HighPass_value, 
                                                "Mincut_Radius"   : self.MinCutRad_value, 
                                                "Maxcut_Radius"   : self.MaxCutRad_value, 
                                                "method"          : self.Method_Value}}


        # X, Y         = self.point_cloud[:, 1], self.point_cloud[:, 0]
        # All_X, All_Y = self.point_All[:, 1], self.point_All[:, 0]

        # coeffs       = Tools.fit_ellipse(np.array(X), np.array(Y))
        # Xc, Yc, a, b, e, PA_LSQE = Tools.cart_to_pol(coeffs)

        # incl           = np.arccos(b/a)
        # PA             = Tools.My_PositionAngle(x0, y0, Yc, Xc, a, b, e, PA_LSQE)
        # X_e, Y_e       = Tools.get_ellipse_pts((Xc, Yc, a, b, e, PA))
        # D              = Tools.Height_Compute(X_e, Y_e, x0, y0)
        # H              = D / np.sin(incl)
        # R              = a
        # Aspect         = H/R

        # First_Ellipse_points = [Xc, Yc, X_e, Y_e, a, b, e, X, Y, All_X, All_Y]
        # First_Estimation     = [incl, R, H, Aspect, 1.219, PA]

        # X_unif, Y_unif = Tools.uniform_ellipse(incl, PA, H, R, 1, R, self.r_beam, x0=x0, y0=y0, init=0)
        # X = []
        # Y = []
        # ny = nx = len(self.image)
        # mapy, mapx = np.indices((ny, nx))
        # for idx in range(len(X_unif)):
        #     mask                  = (mapx - X_unif[idx]) ** 2 + (mapy - Y_unif[idx]) ** 2 <= self.r_beam ** 2
        #     intensities_in_circle = self.image[mask]
        #     max_index             = np.argmax(intensities_in_circle)
        #     mask_indices          = np.where(mask)
        #     max_y, max_x          = mask_indices[0][max_index], mask_indices[1][max_index]
        #     X.append(max_y)
        #     Y.append(max_x)
        # X = np.array(X)
        # Y = np.array(Y)

        # coeffs                   = Tools.fit_ellipse(X, Y)
        # Xc, Yc, a, b, e, PA_LSQE = Tools.cart_to_pol(coeffs)
        # incl                     = np.arccos(b/a)
        # PA                       = Tools.My_PositionAngle(x0, y0, Yc, Xc, a, b, e, PA_LSQE)
        # X_e, Y_e                 = Tools.get_ellipse_pts((Xc, Yc, a, b, e, PA))
        # D                        = Tools.Height_Compute(X_e, Y_e, x0, y0)
        # H                        = D / np.sin(incl)
        # R                        = a
        # Aspect                   = H/R
        # Ellipse_points           = [Xc, Yc, X_e, Y_e, a, b, e]

        # Point_Offset = []
        # for idx in range(len(X)):
        #     idx_dist     = np.argmin(np.sqrt((X_e - X[idx])**2 + (Y_e - Y[idx])**2))
        #     ellipse_dist = np.sqrt((Xc - X_e[idx_dist])**2 + (Yc - Y_e[idx_dist])**2)
        #     point_dist   = np.sqrt((Xc - X[idx])**2        + (Yc - Y[idx])**2)
        #     dist         = ellipse_dist - point_dist
        #     Point_Offset.append(dist)
    
        # Inclinations    = []
        # PositionAngles  = []
        # Heights         = []
        # Mid_Radius      = []
        # for step in range(self.nb_steps):
        #     self.progress.setValue(step + 1)
        #     nb_points = 100
        #     Radius       = np.random.normal(R, np.std(Point_Offset), nb_points)
        #     Phi          = np.random.uniform(0, 2*np.pi, nb_points)
        #     # Xrand, Yrand = Tools.Random_Ellipse(Radius, Phi, x0, y0, incl, R, H, 1, PA)
        #     Xrand, Yrand = Tools.ellipse(incl, PA, H, R, 1, Radius, Phi, x0=x0, y0=y0)
        #     coeffs = Tools.fit_ellipse(np.array(Xrand), np.array(Yrand))
        #     Xc, Yc, a, b, e, PA_LSQE = Tools.cart_to_pol(coeffs)
        #     D_inclinations = np.arccos(b/a)
        #     Inclinations.append(D_inclinations)
        #     PositionAngles.append(Tools.My_PositionAngle(x0, y0, Yc, Xc, a, b, e, PA_LSQE))

        #     D_D            = Tools.Height_Compute(X_e, Y_e, x0, y0)
        #     D_H            = D_D / np.sin(D_inclinations)
        #     Heights.append(D_H)
        #     Mid_Radius.append(a)
        # D_incl   = np.std(Inclinations)
        # D_PA     = np.std(PositionAngles)
        # D_H      = np.std(Heights)
        # D_R      = np.std(Mid_Radius)
        # D_Chi    = 0
        # D_Aspect = Aspect * np.sqrt((D_R/R)**2 + (D_H/H)**2)

        # Data_to_Save    = { "params"        : [incl, R, H, Aspect, 1.219, PA],      # Based on Avenhaus et al. 2018         and         Kenyon & Hartmann 1987 
        #                     "first_estim"   : First_Estimation,
        #                     "Err"           : [D_incl, D_R, D_H, D_Aspect, D_Chi, D_PA], 
        #                     'Points'        : [X, Y, All_X, All_Y], 
        #                     "Ellipse"       : Ellipse_points,
        #                     "first_ellipse" : First_Ellipse_points,
        #                     "Data"          : self.image,
        #                     "Filters"       : {"gaussian_filter" : self.gaussian_value, 
        #                                        "smooth_filter"   : self.smooth_value, 
        #                                        "prominence"      : self.Prominence_value,
        #                                        "distance"        : self.Distance_value, 
        #                                        "width"           : self.Width_value, 
        #                                        "HighPass"        : self.HighPass_value, 
        #                                        "Mincut_Radius"   : self.MinCutRad_value, 
        #                                        "Maxcut_Radius"   : self.MaxCutRad_value, 
        #                                        "method"          : self.Method_Value}}

        with open(f"{self.folderpath}/DRAGyS_Results/{Fit_Name}", 'wb') as fichier:
            pkl.dump(Data_to_Save, fichier)
        self.Filtering_Fig.clf()
        self.Filtering_Canvas.flush_events()
        plt.close(self.Filtering_Fig)
        self.accept()

class Launcher:
    """
    A class to launch the main PyQt Window of DRAGyS
    """
    def Run():
        app = QApplication(sys.argv)
        ex = DRAGyS()  # Instancie la fenêtre principale
        ex.show()
        app.exec()  # Démarre la boucle événementielle
