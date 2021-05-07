# Human-tracking

## Description 
This project aims to combine three methods including Yolo, PCB and Siammask to acheive human tracking.
Note that this project is an application-oriented project. We did not change the original algorithm, but adopted the author’s pre-training model and improved it with actual application requirements.

On the basis of the original siammask implementation of object tracking, these codes can realize that there is no need to manually select the tracking target, only a query picture of a person needs to be input, and the person with the highest match in the video is automatically identified and tracked.

## File structure
- cfg (model configuration file)
- models (pre-trained models)
- utils (some public functions)
- tools (some functions that implement certain specific functions)
- roi_lib (stores　the cache files)
- match_history (stores the visual results of PCB matching during the current video processing)
- output_video (default video output path)

## Environment installation

This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, GTX960M

Download the entire code and switch the working path to the root directory of the code：
```
cd human-tracking
```
Setup python environment:
```
conda create -n human-tracking python=3.6
source activate human-tracking
pip install -r requirements.txt
bash make.sh
```
Add the project to your PYTHONPATH:
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

Finally, before running the demo, you need to download "cfg" and "models" folder from the following links and replace these two folder in ./human-tracking/ directory:

https://drive.google.com/drive/folders/1-cxxqK9lYUKd9GVFlPR_Za5M3cz5kBU-?usp=sharing

https://drive.google.com/drive/folders/1-cxxqK9lYUKd9GVFlPR_Za5M3cz5kBU-?usp=sharing


## DEMO

Before run demo.py, make sure you have setuped your environment.

Demo.py has two required parameters: --query_img and --video：
--query_img sets the path of your query image 
--video sets the path of the video to be processed

For example, my query image and video is in the root directory, named demo_query.jpg and demo_video.mp4, so I run the commit:
```
python demo.py --input demo_query.jpg --video demo_video.mp4
```
After running the demo, you can check your output in the "output" folder.

PS: If the performance of the model is bad on your video, you can adjust the threshold parameter in demo.py.
