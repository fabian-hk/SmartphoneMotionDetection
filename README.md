# Description
In this project i have trained a convolutional neural network
to classify the state of a smartphone into the states: walk, 
bike, rail travel, car drive. To create a data set i have
programed a Android app which records the accelerometer and
the gyroscope while doing this activities. Later on i also
integrated the trained model to make live predictions on
the smartphone. I made this project in spring 2018 in my 
free time besides my computer science studies. 
Here you can see a screenshot from the app:

![Android app](doc/SensorCollectorScreenshot.jpg)

# Training
To train the neural network i collected from every class
roughly 1000 samples. After 15 epochs i have a accuracy
of 98.44% on previously unseen data. You can see the
process of training in the following graphs:

![Training process](doc/TrainingProcess.png)

### Training output

````commandline
Number of classes: 4
Class 0: 1047
Class 1: 1080
Class 2: 1020
Class 3: 1058

Train data: 3365
Val data: 632
Test data: 200

Could not find old network weights

Epoch 0, validation accuracy 0.6516 Time: 2.4834
Epoch 1, validation accuracy 0.7078 Time: 1.1336
Epoch 2, validation accuracy 0.8016 Time: 1.1266
Epoch 3, validation accuracy 0.9359 Time: 1.1346
Epoch 4, validation accuracy 0.8797 Time: 1.1167
Epoch 5, validation accuracy 0.8906 Time: 1.1256
Epoch 6, validation accuracy 0.875 Time: 1.1482
Epoch 7, validation accuracy 0.9203 Time: 1.165
Epoch 8, validation accuracy 0.9828 Time: 1.1248
Epoch 9, validation accuracy 0.975 Time: 1.1356
Epoch 10, validation accuracy 0.9781 Time: 1.1366
Epoch 11, validation accuracy 0.9781 Time: 1.1616
Epoch 12, validation accuracy 0.9781 Time: 1.1486
Epoch 13, validation accuracy 0.9797 Time: 1.1396
Epoch 14, validation accuracy 0.9797 Time: 1.1822
Final test accuracy 0.984375
````


