# pose-prediction
Start by cloning this repositiory:

git clone https://github.com/vita-epfl/pose-prediction.git
cd pose-prediction

Create a new conda environment (Python 3.7):

conda create -n pose-de python=3.7
conda activate pose-de

And install the dependencies:

pip install -r requirements.txt

    The following libraries are required to run this code: Pytorch 1.4.0+, OpenCV, Numpy, Pandas, PIL, matplotlib, glob, json

    
Usage
python train.py
by running python train, you will train the model for the posetrack data set. During the training it always saves the best (based on vim_train) model parameters and output files for somof submission in models/ and submission/ respectively.



Results can be visualized during training too. The code needs to be uncommented. 

Tested Environments:

    izar

