# Hand Writing RNN

This RNN is a reconstruction of Alex Graves' work presented in his paper: http://arxiv.org/abs/1308.0850

## Model

The model is precisely the same as the one presented in the paper above. Essentially, this is an RNN with three
LSTM layers, a windowing layer that learns attention for each character in the input string, and a mixture of
Gaussians criterion layer.

To train the model:
```
th driver.lua 
```

Explore driver.lua for command options. Including specifying a validation and training set. Currently toy.t7 is provided, but a larger training set can be created by downloading the IAM handwriting database and following the directions in the Graves' paper: http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database

## Results

* Each multicolored line represents a new pen stroke.

```
th testNet.lua -string 'How are you today'
```
![alt tag](https://github.com/jarmstrong2/handwritingnet/blob/master/samples/howareyoutoday.png)

```
th testNet.lua -string 'Testing testing'
```
![alt tag](https://github.com/jarmstrong2/handwritingnet/blob/master/samples/testingtesting.png)

```
th testNet.lua -string 'this is awesome'
```
![alt tag](https://github.com/jarmstrong2/handwritingnet/blob/master/samples/thisisawesome.png)

* Below is the attention plot which represents by how many timesteps the model will focus on a certain character in a string (moving in a left to right direction)

![alt tag](https://github.com/jarmstrong2/handwritingnet/blob/master/samples/thisisawesome_attention.png)
