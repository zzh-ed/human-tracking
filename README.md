# Human-tracking

## Description 

### Background
SiamMask is the current advanced target tracking algorithm, but this algorithm has limitations: before each tracking, the tracking target needs to be artificially defined, and the target cannot be reacquired after the target is lost. So this project aims to combine three methods including Yolo, PCB and Siammask to acheive human tracking without the manual initialization.

### Implement
The complete workflow of the system is shown:


To deal with the manual initialization, I implement two new models, Yolo and PCB model, into the SiamMask model. For the YOLO model, I first set the target as person, and the Yolo Model can identify and segment the people in the that frame of video, and generate an image gallery. After the image gallery is created, the PCB model will compare the query image with all the images in the gallery for feature mapping, and return the image that both scores greater than the initial threshold and scores the highest in similarity as the target object. If there is no object that meets this condition, this frame will be skipped, and the same processing will be performed on the next frame until the target object is identified. This is a solution to automatically initialize SiamMask tracking without manually framing out the target.
To deal with the target loss problem, I decide to set the threshold for the SiamMask tracking accuracy. SiamMask threshold means that the value below which the target is considered as lost and the target needs to be reinitialized, when SiamMask is tracking the target. When SiamMask tracks a target, it returns the similarity score of the target between frames. When this score decreases, it means that the target disappears or is occluded. Therefore, I set a SiamMask threshold. If the score is greater than the SiamMask threshold, the model will continue tracking and update the query image as the latest one. Otherwise, if the accuracy is lower than the particular threshold, meaning that the SiamMask may be tracking the wrong target, it will stop tracking and initialize again with the last updated query image.

### Robustness 
To ensure the robustness of the extended SiamMask model, I use the dynamic update method for threshold setting and the real-time update of query pictures method. In the process described above, the PCB initial threshold and SiamMask threshold are used. The PCB initial threshold refers to how much the score of the image in the gallery with the highest score needs to be reached when using PCB to recognize the target. If the PCB initial threshold is too high, the true target will be filtered out. If it is too low, it will cause other objects to be mistakenly regarded as targets when the target disappears. Therefore, here I use two initial thresholds. At the beginning of the video, the gap between the query image and the target will be relatively large, so I set a lower threshold. When the target is tracked, the model will continuously update the query picture with the currently tracked target. Then the PCB is used to match target the object, and the updated query image is used for matching. Since the gap between the query image and the true target reduces, I use a higher threshold to ensure the accuracy of target recognition. When SiamMask is tracking a target, it will return the similarity score of the target between frames. When this score decreases, it means that the target disappears or is occluded. Therefore, I set a SiamMask threshold for the model. When the score is lower than the SiamMask threshold, the SiamMask will stop working, and YOLO and PCB will be called to re-identify the target object.
Therefore, the robustness of the extended model majorly depends on three parameters: an initial threshold for the PCB model, a reinitialized threshold for the PCB model and a threshold for the SiamMask model.

## Output example
On the basis of the original siammask implementation of object tracking, these codes can realize that there is no need to manually select the tracking target, only a query picture of a person needs to be input, and the person with the highest match in the video is automatically identified and tracked.

Here is an output example of these code:

The query image:

<img src="https://github.com/zzh-ed/human-tracking/blob/master/demo_query.jpg" width="105" height="305" /><br/>
The output video:
![img](https://github.com/zzh-ed/human-tracking/blob/master/demo_output.gif)

## File structure
- cfg (model configuration file)
- models (pre-trained models)
- utils (some public functions)
- tools (some functions that implement certain specific functions)
- roi_lib (stores the cache files)
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

https://drive.google.com/drive/folders/1oD-vzdxBapUom-QFCfBS-1abNfoHp11P?usp=sharing


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


## Reference:
- [1] https://github.com/foolwood/SiamMask
- [2] https://github.com/layumi/Person_reID_baseline_pytorch
- [3] https://pjreddie.com/darknet/yolo/

