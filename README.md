# Overview
This project applies the method of double operations for deepfake detection.
The work has been submitted to IEEE Signal Processing Letters.
This project reuses pre-trained models and codes from [icpr2020dfdc](https://github.com/polimi-ispl/icpr2020dfdc).

# Dataset
The dataset can be found at [this link](https://drive.google.com/drive/u/1/folders/1QhAOZRXO0_PN-DkKbUCkzjM6WfXLaE_m).
The raw and deepfake videos of ten public figures are saved in the folder 'celeb_videos'.
The reconstructed videos of the raw and deepfake videos can abe found in the folder 'reconstructed_videos'.
Before running the experiments, the videos should be downloaded.

# Usage
To save computational time, the file [prepare features.ipynb](https://github.com/sevprinc/DF/blob/main/notebook/prepare%20features.ipynb) is first used to extract features from the videos before and after reconstruction using a pretrained neural network. 
The extracted features will be used to train the Siamese neural network for deepfake detection.
Then, the file [Double operation (Siamese NN).ipynb](https://github.com/sevprinc/DF/blob/main/notebook/Double%20operation%20(Siamese%20NN).ipynb) can be used to customize a Siamese neural network to detect deepfake videos of each public figure using the proposed double-operation method.


