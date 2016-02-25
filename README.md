# Hand Writing RNN

This RNN is a reconstruction of Alex Graves' work presented in his paper: http://arxiv.org/abs/1308.0850

## Results
* Each multicolored line represents a new pen stroke.

```
th testNet.lua -string 'How are you doing today'
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
