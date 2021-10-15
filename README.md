# Mask Detector 

## DEPLOY: command line 
conda create --name mask_check --clone base
conda activate mask_check
pip install -r requirements.txt

##model for mask detection
with tf.keras built-in MobileNetV2 pretrained model

## USAGE
train.py: for model training
move pretrained model into exported directory and then run webcam_demo.py script

## NOTE 
HUNGER for training data for improving model accuracy and lasted model are easily detected when masking by other objects (human hands, etc. )
hung
