import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
import subprocess
import shutil
import pathlib

current_folder = pathlib.Path(__file__).parent
print(f"Le chemin d'accès du fichier Python actuel est : {current_folder}")
GUI_Folder     = os.path.join(current_folder, "DRAGyS")
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