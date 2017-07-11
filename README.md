# Pupil tracking algorithm software implementation and hardware FPGA-based acceleration for identification and evaluation of Traumatic Brain Injury

**Authors:** Bracco Filippo and Di Vece Chiara 

**Email:** filippo.bracco@mail.polimi.it, chiara.divece@mail.polimi.it

## Brief description of project:
Concussion is the leading cause of death among young people under 40 years in industrialized  countries. Due to this high incidence it is fundamental to optimally diagnose this disease.
Aim of the project is accelerate on FPGA of an OpenCV application for pupillometry measurement, in order to help the neurological assessment. Thanks to the pupil detection and tracking, the pupillometer allows to estimate pupil diameter and  reactivity to light (photopupillary  reflex) in a much more accurate and faster way than a doctor's human eye can do.This will provide accurate values to evaluate the seriousness of a Traumatic Brain Injury (TBI).
OpenCV application is executed and accelerated on a PYNQ-Z1 board, using Python 3 and a Jupyter notebook.


## Description of archive (explain directory structure, documents and source files):

/Software Implementation/ - location of video acquisition, full software implementation source code 

-> /Assets/ - needed assets with also a sample of aquired video 
-> /C++/ - C++ projects both unoptimized and using optimized search
-> /Python/ - first draft application in Python 3

/Hardware Implementation/ - location of full hardware accelerated application source code 

-> /IP/ - contains Vivado HLS core sources (including c code)
-> /overlay/ - contains  overlay of implemented design
-> /wrapper/ - contains the python module which allows to call hardware accelerator
-> /wrapper/test/ - an example of how to use module, with also a image of eye

/Data Analysis/ - location of MATLAB source code used to analyse and plot data


## Instructions to build and test the project

### Software Implementation

**Third parties:**

* OpenCV (http://opencv.org)
* Intel SDK (https://software.intel.com/en-us/intel-realsense-sdk/download) (https://downloadmirror.intel.com/25044/eng/sr300_3_3_release_notes.pdf) 

Once installed third parties application can be executed; be sure that Assets folder path is correct in source code.

### Hardware Implementation

**Board used:** PYNQ-Z1

**Vivado Version:** 2016.1

##### Prerequisites:
* PYNQ-Z1 board
* Computer with compatible browser
* Ethernet cable 
* Micro-SD card with preloaded PYNQ-Z1 image (download here: http://www.pynq.io)

* *Step 1:* Setup the device (detailed guide here: https://pynq.readthedocs.io/en/latest/1_getting_started.html#prerequisites)
* *Step 2:* Copy into Pynq/Overlay both files in hw folder (biepincs.bit and biepincs.tcl)
* *Step 3:* Copy Assets folder and Application.ipynb in the same path on Pynq; open application with jupyter and start kernel.

If you want to use hardware accelerator in another application: copy wrapper_imagefilter.py module (it is in /Hardware Implementation/wrapper/) into Pynq folder and import it in your application. Example in /Hardware Implementation/test/ shows the use of module.
For more detailed information about hardware design go to (https://bitbucket.org/necst/xohw17_bie-pincs_public)


## References:

**Project Slideshare page:** https://www.slideshare.net/BIEPInCS

**Project repository for detailed hardware documentation:** https://bitbucket.org/necst/xohw17_bie-pincs_public

**Project YouTube playlist:** https://www.youtube.com/playlist?list=PLewc2qlpcOudk0TcxfQjREWd_wpiFVJA2

