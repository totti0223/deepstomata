
# DeepStomata (stomatal pore quantifier)
### Facial Recognition Technology for Automated Stomatal Aperture Measurement
At a glance

<img src="https://github.com/totti0223/deepstomata/blob/master/images/ataglance.jpg" width="480">


A three step image analysis program for quantification of stomatal aperture from bright field images.

1. Identifying the coordinate of the stomata by HOG + SVM.
2. Classifying the status (open, partially open, closed, false positive) by CNN.
3. Pore quantification responsive to the object status.

<img src="https://github.com/totti0223/deepstomata/blob/master/images/main.jpg" width="600">

# Author
Yosuke Toda, Ph.D (Agriculture)

tyosuke@aquaseerser.com

Post Doctoral Researcher

Lab of Plant Physiology

Department of Science

Nagoya University, Japan

## Requirements

python>3

matplotlib==1.5.1

numpy==1.11.2

scipy==0.18.1

scikit_image==0.12.3

tensorflow==0.10.0rc0

Pillow==3.2.0

common==0.1.2

cv2==1.0

dlib==19.1.0

setuptools==32.3.1

## Installation

1. Download this repository.

2. Unzip.

2. Open terminal.

3. Move to the Unzipped directory.

~~~~
pip install .
~~~~

## Note

- Tensorflow must not be ver. 1.0.. Codes are not compatible.

- Several packages such as cv2 and dlib cannot be installed via pip in anaconda environment. In such cases, comment out the requirements.txt like the following 

~~~~
#cv2 ==1.0
#dlib == 19.1.0
~~~~
and install respectively via conda install

## Usage

- In terminal

~~~~
python
import deepstomata
deepstomata.cui("PATH/TO/THE/DIRECTORY_OR_IMAGES")
~~~~

## Example

### 1
Analyze a directory containing 4 jpeg images in the example folder

~~~~
import bmicp
bmicp.cui("PATH_TO_THE_EXAMPLE_FOLDER/examples")
~~~~

![ex1](https://github.com/totti0223/stomata_quantifier/blob/master/images/e1.png)

### 2
Result overview in the terminal.

![ex2](https://github.com/totti0223/stomata_quantifier/blob/master/images/e2.png)

### 3

#### Result directory overview.
![ex3](https://github.com/totti0223/stomata_quantifier/blob/master/images/e3.png)

#### annotated/IMAGE_stomata_position.jpg
Image with stomata position.

#### annotated/IMAGE_stomata_tiled.jpg
Montage image of stomata candidate that is same size as input image.

#### annotated/IMAGE_classified.jpg
Image with stomata position, class percentage.

#### annotated/IMAGE_all.jpg
Image with stomata position, class percentage, and segmented stomatal pore.

#### FOLDERNAME_all.csv
CSV files with quantified stomatal pores.

#### FOLDERNAME_count.csv
Statistics of classified object per image (no. of open, partially open, closed, false positive per image).

## Plans

- Migrating CNN code from tensorflow to keras

- GUI

- Registrating the package to PyPi (Packaging the CNN model exceeds the upload size limit of PyPI)