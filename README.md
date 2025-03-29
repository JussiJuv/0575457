# Setup instructions

## 1. Install OpenPose from this link:

https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases

Version OpenPose v1.7.0 (GPU) was used in this project.

## 2. Change the testing file paths in two files

### 2.1 test_process_skeletons.bat

Change OPENPOSE_PATH to where you installed OpenPose.\
Change BASE_INPUT to where the testing images are located. Make it the root folder where folders "1", "2",... "32" are located.\
Change BASE_OUTPUT to where the skeletons will be saved. Recommended to use the empty testing_skeleton folder in this project.

### 2.2 config.py

Change TEST_DIR, same as BASE_INPUT.\
Change TEST_SKELETON_ROOT, same as BASE_OUTPUTS.

### 2.3 testing_skeletons folder
you probably should delete the .gitkeep file here, this was added so I could commit an empty folder. If kept, might break functionality, unlikely though.

## 3. Run test_process_skeletons.bat

This should process all the skeletons and create 32 folders inside testing_skeleton.\
Individual keypoints should be then stored like this "\0575457\testing_skeleton\1\json\clipID.frameID_keypoints.json"

## 4. Create a Virtual Environment

```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
```

## 5. Install the requirements (Modify if necessary / check your cuda version with nvcc --version)

### Modify to install the correct PyTorch version based on your CUDA version.

For example for CUDA12.1, replace cu118 with cu121\
Alternatively you can change the requirements.txt and modify the torch and torchvision versions.\
However the code is only tested with the versions in requirements.txt.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

## 6. Run python inference.py

This file has the example inference as requested in the assignment.\
Below is an example output when executing

```bash
python inference.py
```

Loading model...\
Model epoch_20.pt loaded!\
Loading test dataset...\
Test Dataset:

- Clips: 2593\
- Skipped frames: 1\
- Clips with all invalid frames: 0\
  Loaded 2593 test samples\
  Evaluation in progress...

Test Results:\
Top-1 Accuracy: 55.15%\
Top-5 Accuracy: 91.21%

### Additional notes

This project was made with RTX 2060 and CUDA 11.8.\
A lot of files were not added to the project to keep it from bloating.\
Returned codes only have necessary files for inference and the training file is also attached.\
The training logs (csv files) are in the folder logs.

training_log.csv - Fusion model\
rgb_training_log.csv - RGB only model\
skel_training_log.csv - Skeleton only model
