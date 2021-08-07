# mAP_calculation
This repo is used for calculating metrics for evaluation of detection systems.  

Especially, the scripts were written for the dataset of **GTSDB** (https://benchmark.ini.rub.de/gtsdb_dataset.html) and prediction outcome in JSON, in which predictions could be found in 'frames' from 'output'. Before running the scripts, all the files in this repo should be draged to folder 'FullIJCNN2013'.  

```pip3 install -r requirements.txt``` to install required libraries.
  
1. ```python3 parseJson.py```  
parseJson.py is used for annotating two new classes in the dataset. There will be an interface helping mannual annotation, as shown below. After running this, gt.txt will be updated.
<img src="https://github.com/NantonZZZ/mAP_calculation/blob/master/interface.jpeg" width="65%"/>

2. ```python3 label.py```  
label.py will create gt files for each images in the dataset for evaluating the prediction outcome.

3. ```python3 mAP.py```  
In this script, recall, precision, AP will be calculated. Details will be shown in csv files. Problematic cases will also be included in a new folder 'FP&FN'.
