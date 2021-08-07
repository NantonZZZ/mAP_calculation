# mAP_calculation
This repo is used for calculating metrics for evaluation of detection systems.  

Especially, the scripts were written for the dataset of **GTSDB** (https://benchmark.ini.rub.de/gtsdb_dataset.html) and prediction outcome in JSON, in which predictions could be found in 'frames' from 'output'. Before running the scripts, all the files in this repo should be draged to folder 'FullIJCNN2013'.
  
1. ```python3 parseJson.py```  
parseJson.py is used for annotating two new classes in the dataset. There will be an interface helping mannual annotation, as shown below. After running this, gt.txt will be updated.
<img src="https://github.com/NantonZZZ/YOLOv3_GTSDB/blob/master/00051.jpg" width="65%"/>

2. 
