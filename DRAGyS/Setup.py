import sys
import os
import subprocess

def install_package(package_name):
    try:
        __import__(package_name)
        print(f"'{package_name}' is already installed.")
    except ModuleNotFoundError:
        print(f"'{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

for package_name in ['pickle', 'PyQt5', 'scipy', 'astropy', "multiprocess"]:
    install_package(package_name)

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
import shutil
import pathlib

current_folder = pathlib.Path(__file__).parent
print(f"Le chemin d'accès du fichier Python actuel est : {current_folder}")
GUI_Folder     = os.path.join(current_folder, "GUI")
Fitting_Folder = os.path.join(current_folder, "Fitting")
SPF_Folder     = os.path.join(current_folder, "SPF")
os.makedirs(GUI_Folder,     exist_ok=True)
os.makedirs(Fitting_Folder, exist_ok=True)
os.makedirs(SPF_Folder,     exist_ok=True)

# Enregistrer le chemin dans un fichier path.txt
path_file = os.path.join(GUI_Folder, "DRAGyS_path.txt")
with open(path_file, "w") as f:
    f.write(GUI_Folder)
    f.write("___separation___")
    f.write(Fitting_Folder)
    f.write("___separation___")
    f.write(SPF_Folder)
for python_file in ["Main.py", "SPF_Window.py", "Filter_Pixel.py", "Tools.py", "Setup.py"]:
    shutil.move(f"{current_folder}/{python_file}", GUI_Folder)
