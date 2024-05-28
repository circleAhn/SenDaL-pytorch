# SenDaL: An Effective and Efficient Calibration Framework of Low-Cost Sensors for Daily Life

<center><img src = "/img/intro.png" width = "70%" height = "70%"></center>

## Abstract
The collection of accurate and noise-free data is a crucial part of Internet of Things (IoT)-controlled environments. However, the data collected from various sensors in daily life often suffer from inaccuracies. Additionally, IoT-controlled devices with low-cost sensors lack sufficient hardware resources to employ conventional deep learning models. To overcome this limitation, we propose sensors for daily life (SenDaL), the first framework that utilizes neural networks for calibrating low-cost sensors. SenDaL introduces novel training and inference processes that enable it to achieve accuracy comparable to deep learning models while simultaneously preserving latency and energy consumption similar to linear models. SenDaL is first trained in a bottom-up manner, making decisions based on calibration results from both linear and deep learning models. Once both models are trained, SenDaL makes independent decisions through a top-down inference process, ensuring accuracy and inference speed. Furthermore, SenDaL can select the optimal deep learning model according to the resources of the IoT devices because it is compatible with various deep learning models, such as long short-term memory-based and Transformer-based models. We have verified that SenDaL outperforms existing deep learning models in terms of accuracy, latency, and energy efficiency through experiments conducted in different IoT environments and real-life scenarios.

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
