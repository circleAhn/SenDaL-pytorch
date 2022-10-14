# SenDaL: A Data-driven Approach to Upgrade Low-cost Sensors for Daily Life

<center><img src = "/img/intro.png" width = "700%" height = "70%"></center>

## Abstract
The development of the Internet of Things (IoT) has gradually revolutionized our lives. The crucial part of IoT-controlled environments is the collection of accurate and clean (noise-free) data. Unfortunately, the existing IoT sensors have hardware limitations in terms of processing, storage, and throughput which limits the accuracy and speed of data collection. The prevalent solutions have suggested several ways to overcome the hardware limitations with deep learning and to collect their data accurately in some areas, such as data-driven robotics and soft sensors. However, studies on upgrading low-cost sensors applied to home appliances in daily life still depend on a simple linear model owing to its limited hardware resources. To fill this necessary research gap, we propose SenDaL (Sensors for Daily Life), an advanced data-driven approach for low-cost sensors. SenDaL uses a linear model with a fast inference speed in gradual time-series changes, while using a deep learning model with high accuracy in rapid time-series changes. Our model can also apply different deep learning models (e.g., LSTM, GRU, and Transformer) to design an optimal SenDaL that is appropriate for specific IoT environments based on the three-step training process. To validate the performance of SenDaL, we collected and refined the fine-dust dataset in diverse real-life scenarios. Our experimental results demonstrated that SenDaL significantly outperformed existing deep learning models in terms of accuracy and inference speed. 

<br />

## Requirements (on machine)
Machine are experiemented on virutal machine on Anaconda 3.
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


<br />

## Datasets
We provide 3 real-world fine-dust datasets, which is refined and syncrhonized between high- and low- cost sensors.
* ```home1.csv```
* ```home2.csv```
* ```home3.csv```


<br />

## Run
We provide 3 running options: SenDaL training, computing accuracy, and computing inference time.

To validate our result, models can be trained using **anchored walk forward optimization** to evaluate SenDaL using 10-fold cross-validation. The performance of each environment was measured as the average from 10 results for each high-accuracy sensor. The overall performance was calculated as the average performance of each environment.

For computing accuracy and inference time, pretrained SenDaL is required. Since the training process takes a long time, we provide all the weight of the pre-trained SenDaL (with LSTM component) trained with anchored walk forward optimization as an example for verification accuracy. All the weight files are formatted as ```lstm_home<x>_sensor<y>_<z>of<n_cv>_<type>.pkl``` where ```x``` denotes environment number, ```y``` denotes sensor number, ```z``` denotes the z-th fold of cross-validation, ```n_cv``` denotes the number of folds, and ```type``` denotes single or unified model.

Other components (e.g., GRU, PLSTM, Transformer) also should be trained, but we do not provide a pre-trained model as above.


### Run (SenDaL training)
```
$ python model_training.py <dataset_name> <sensor_column> <model_name> <n_cv>
```

* Initial ```<dataset_name>``` format are: ```home1.csv, home2.csv, home3.csv```. 
* Initial ```<sensor_column>``` format are: ```sensor1, sensor2, sensor3, sensor4```. 
* Valid ```<model_name>``` format are: ```LSTM, GRU, PhasedLSTM(or PLSTM), Transformer(or Trans)```. 
* Default of ```<n_cv>``` is set to ```10```. 

For example, we want to train an lstm-SenDaL model using dataset home1 with sensor1 based on 5-fold cross-validation, the training command is:
```
$ python model_training.py home1.csv sensor1 lstm 5
```

Then model_training.py will create ```2*n_cv``` files which is of the form:

```<model_name>_home<x>_sensor<y>_<z>of<n_cv>_single.pkl``` and ```<model_name>_home<x>_sensor<y>_<z>of<n_cv>_unified.pkl```


**Note**: To verify the results of the paper, the average is calculated by learning a 10-fold cross-validation for all ordered parirs of dataset and sensor. Since the training process takes a long time and often sensitive, we provide all the weight of the pre-trained SenDaL (with LSTM component) trained with anchored walk forward optimization as an example for verification accuracy. We can easily show the experimental results using the following commands.



### Run (Checking pretrained model accuracy)
```
$ python model_eval_accuracy.py <model_name> <n_cv> <type>
```
* Valid ```<model_name>``` format are: ```LSTM, GRU, PhasedLSTM(or PLSTM), Transformer(or Trans)```. 
* Default of ```<n_cv>``` is set to ```10```. ```<n_cv>``` must equal to pretrained models setting.
* ```<type>```: ```1``` is for checking single model, and ```2```is for checking unified model. Different models are needed for each type. To fit our experimental results, 120 (3x4x10) pre-trained weights are required for each type.

**Note**: **Pretrained models must be required.** 

The following two commands are available without training:
```
$ python model_eval_accuracy.py lstm 10 1

output:
  ...
  Home1:  2.233357253319351
  Home2:  2.3198059027357276
  Home3:  3.0535407238081946
  Overall:  2.5621276964843593
```
and
```
$ python model_eval_accuracy.py lstm 10 2

output:
  ...
  Home1:  2.1954087284152037
  Home2:  2.297551276696442
  Home3:  3.041442506113538
  Overall:  2.5396150542844946
```


### Run (Checking pretrained model inference speed)
```
$ python model_eval_time.py <model_name> <n_cv> <type>
```
* Valid ```<model_name>``` format are: ```LSTM, GRU, PhasedLSTM(or PLSTM), Transformer(or Trans)```. 
* Default of ```<n_cv>``` is set to ```10```.  ```<n_cv>``` must equal to pretrained models setting.
* ```<type>```: ```1``` is for checking single model, and ```2```is for checking unified model. Different models are needed for each type. To fit our experimental results, 120 (3x4x10) pre-trained weights are required for each type.

**Note1**: **Pretrained models must be required.**

**Note2**: This code is focused on CPU inference time. If we check the inference time on GPU invironment, different time measurement method is required. Change to :
```python
# in Models.py

...
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
...
with torch.no_grad():
  for rep in range(repetitions):
    ...
    starter.record()
    ... # predict
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)/(len(testX))
    ...
...
```
<br />

## Experimental results

### Results of correction accuracy (RMSE)

<center><img src = "/img/res1.png" width = "70%" height = "70%"></center>

### Results of inference speed on machine (ms, us)

<center><img src = "/img/res2.png" width = "70%" height = "70%"></center>

### Results of inference speed on embedded hardware (us)

<center><img src = "/img/res3.png" width = "70%" height = "70%"></center>

<br />

## Contact

Seokho Ahn (sokho0514@gmail.com)
