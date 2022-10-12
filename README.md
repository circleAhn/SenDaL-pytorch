# SenDaL: A Data-driven Approach to Upgrade Low-cost Sensors for Daily Life

<center><img src = "/intro.png" width = "50%" height = "50%"></center>

## Abstract
The development of the Internet of Things (IoT) has gradually revolutionized our lives. The crucial part of IoT-controlled environments is the collection of accurate and clean (noise-free) data. Unfortunately, the existing IoT sensors have hardware limitations in terms of processing, storage, and throughput which limits the accuracy and speed of data collection. The prevalent solutions have suggested several ways to overcome the hardware limitations with deep learning and to collect their data accurately in some areas, such as data-driven robotics and soft sensors. However, studies on upgrading low-cost sensors applied to home appliances in daily life still depend on a simple linear model owing to its limited hardware resources. To fill this necessary research gap, we propose SenDaL (Sensors for Daily Life), an advanced data-driven approach for low-cost sensors. SenDaL uses a linear model with a fast inference speed in gradual time-series changes, while using a deep learning model with high accuracy in rapid time-series changes. Our model can also apply different deep learning models (e.g., LSTM, GRU, and Transformer) to design an optimal SenDaL that is appropriate for specific IoT environments based on the three-step training process. To validate the performance of SenDaL, we collected and refined the fine-dust dataset in diverse real-life scenarios. Our experimental results demonstrated that SenDaL significantly outperformed existing deep learning models in terms of accuracy and inference speed. 



## Requirements (Machine)
* ```Python 3.8.5.```
* ```Pytorch 1.8.0.```
* ```Pandas 1.2.4.```
* ```Numpy 1.19.5.```

Embedded hardwares (Jetson Nano, Raspberry Pi 3, Raspberry Pi 4) are different versions of the environment than machine, but all of the above libraries should be imported. The inference speed of the model may vary depending on the environment.

#### Machine environment (Both trained and predicted)
* CPU: AMD Ryzen 5 5600X 6-Core
* GPU: NVIDIA GeForce RTX 2060

#### Embedded hardware environment (Only predicted using pretrained SenDaL)
* Jetson Nano (CPU): 1.43GHz 4-core ARM Cortex-A57
* Raspberry Pi 3 (CPU): 1.4GHz 4-core ARM Cortex-A53
* Raspberry Pi 4 (CPU): 1.5GHz 4-core ARM Cortex-A72

Jetson Nano also have GPU, but it was not considered in our experiment.




## Datasets
We provide 3 real-world fine-dust datasets, which is refined and syncrhonized between high- and low- cost sensors.
* ```home1.csv```
* ```home2.csv```
* ```home3.csv```


## Run
We provide 3 running options: SenDaL training, computing accuracy, and computing inference time.

To validate our result, models can be trained using **anchored walk forward optimization** to evaluate SenDaL using 10-fold cross-validation. The performance of each environment was measured as the average from 10 results for each high-accuracy sensor. The overall performance was calculated as the average performance of each environment.

For computing accuracy and inference time, pretrained SenDaL is required. Since the training process takes a long time, we provide all the weight of the pre-trained SenDaL (with LSTM component) trained with anchored walk forward optimization as an example for verification accuracy. All the weight files are formatted as ```lstm_home<x>_sensor<y>_<z>.pkl``` where x denotes environment number, y denotes sensor number, and z denotes the z-th cross-validation.

Other components (e.g., GRU, PLSTM, Transformer) also should be trained, but we do not provide a pre-trained model as above.


### Run (SenDaL training)
~~```$ python3 model_training.py <model_name>```~~

Command is not yet available.

Available ```model_name``` are: ```LSTM, GRU, PLSTM, Transformer```.


### Run (Checking pretrained model accuracy)
~~```$ python3 model_predicting.py <model_name> <pretrained_model>```~~

Command is not yet available.


### Run (Checking pretrained model inferencing)
~~```$ python3 model_train_eval.py <model_name> <pretrained_model>```~~

Command is not yet available.

