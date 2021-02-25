# Faster-RCNN-with-torchvision
Build Faster-RCNN through the modules officially provided by pytorch torchvision for detection and learning.


## Installation
##### Code and environment construction
    $ git clone https://github.com/supernotman/Faster-RCNN-with-torchvision.git
    $ cd Faster-RCNN-with-torchvision/
    $ sudo pip install -r requirements.txt

##### data preparation
The current code only supports training of the coco dataset
1. Download the coco2017 dataset

2. The data set file structure after downloading is as follows:

```Shell
  coco/
    2017/
      annotations/
      test2017/
      train2017/
      val2017/
```

## Training and testing
##### command
```
python -m torch.distributed.launch --nproc_per_node=6 --use_env train.py --world-size 6 --b 4
```

##### Partial parameter description
```
[--nproc_per_node]     
[--b]                  
[--epochs]            
[output-dir]          
``` 

## Single image detection
```
$ python detect.py --model_path result/model_13.pth --image_path imgs/1.jpg
```

## result

##### AP
IOU | area |  maxDets  |  value    
-|-|-|-
0.50:0.95 | all | 100 | 0.352    
0.50 | all | 100 | 0.573   
0.75 | all | 100 | 0.375   
0.50:0.95 | small | 100 | 0.207 
0.50:0.95 | medium | 100 | 0.387 
0.50:0.95 | medium | 100 | 0.448 

##### AR
IOU | area |  maxDets  |  value    
-|-|-|-
0.50:0.95 | all | 1 | 0.296  
0.50:0.95 | all | 10 | 0.474  
0.50:0.95 | all | 100 | 0.498  
0.50:0.95 | small | 100 | 0.312  
0.50:0.95 | medium | 100 | 0.538  
0.50:0.95 | medium | 100 | 0.631  

##### Example result
<p align="center"><img src="assets/9.jpg" width="320"\></p>
<p align="center"><img src="assets/4.jpg" width="320"\></p>