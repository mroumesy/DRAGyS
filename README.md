
# *DRAGyS*

*DRAGyS* stands for Disk Ring Adjustment Geometry yields to Scattering phase function. This project aims to extract the Scattering Phase Function (SPF) from protoplanetary disk images. 
Unlike traditional approaches that rely on prior time-consuming disk modeling, *DRAGyS* directly analyzes disk images to estimate their geometry and extract the SPF. 
It also considers geometric effects, like limb brightening, which are often neglected but can significantly impact the derived SPF. This tool represents a significant step forward in studying SPFs from ring-shaped protoplanetary disks.

# Download and Install

You can download *DRAGyS* via

    git clone https://github.com/mroumesy/DRAGyS.git
if you want to use *DRAGyS* as package _dragys_, enter in _DRAGyS\python_ directory

    cd DRAGyS/python/
and run 

    pip install -e .
In a python file, you can now import _dragys_ and launch the GUI. Typically, enter these python lines to lauch the tool

    import dragys
    if __name__ == "__main__":
        dragys.Launcher.Run()
line *\_\_name\_\_ == "\_\_main\_\_":* ensures that the window will not open multiple times due to multiprocessing during SPF computation.
The tool first asks you to define a backup folder. It will create a “DRAGyS_Results” folder to store the fitting and SPF files of the disks analyzed.
# Attribution

If you would like to publish results generated by using *DRAGyS*, please cite the following paper.
 - Roumesy et al. 2025

# License
Copyright 2025 Maxime Roumesy

*DRAGyS* is distributed under the MIT License. See the LICENSE file for the terms and conditions.
