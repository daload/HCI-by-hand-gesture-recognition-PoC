<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
<h3 align="center">RACING GAME CONTROLLER BY IMAGE-BASED HAND GESTURE RECOGNITION (PoC)</h3>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

This project was done as a proof of concept to support my undergraduate thesis. This code was a complement of the 
project to exemplify how a racing game controller by image-based hand gesture recognition can be done.
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may control a racing game with hand gestures thanks to an image input and a CNN.

### Prerequisites

* Install requirements:
  ```sh
  pip install -m requirements.txt
  ```

### Installation

1. Clone the repo
2. Install packages

<p align="right">(<a href="#top">back to top</a>)</p>

### Execution

Since this PoC covers 3 different approaches:
  ```sh
  python main.py -v {1, 2 or 3 to choose the approach}
  ```

<p align="right">(<a href="#top">back to top</a>)</p>

### Configuration

* Confidence threshold: In order to change the threshold of confidence of the CNN head to the “constants.py” file 
  located in the root and modify the “threshold” variable
* Gesture mappings: To change the gesture mappings of the 1st and 2nd approach head to the “constants.py” file located 
  in the root and modify the values of the “gestures” dictionary. 
  
  To change the 3rd approach mappings, change the “mappings” dictionary keys in the “v3.py” file.
* Angle needed to turn: To change the maximum angle to turn in the 2nd and 3rd approach head to either “v2.py” or 
  “v3.py” depending on the one you want to change. Changing “v2.py” will change the 2nd approach angle while changing 
  “v3.py” will affect to the 3rd approach angle. Modify the “turning_angle” variable. 30 means that generating an angle 
  of 30 degrees you will get the maximum turning input

<p align="right">(<a href="#top">back to top</a>)</p>

### Recommendations
These are a couple of recommendations for using this PoC:
* First, stay a bit away of the camera. If you are close is more probable that by moving the hand some part 
  will be out of the frame, and the gesture will not be recognized.
* Test that the gestures you selected before playing seriously. Might happen that because of a complex background some 
  gestures are not detected properly. For example, “palm” and “stop” worked fine for all cases, but the “fist” gesture 
  had problems depending on the background.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [A. Kapitanov, A. Makhlyarchuk, and K. Kvanchiani](https://github.com/hukenovs/hagrid)

<p align="right">(<a href="#top">back to top</a>)</p>