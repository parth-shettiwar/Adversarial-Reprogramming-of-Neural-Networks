# Adversarial-Reprogramming-of-Neural-Networks
Implementation of paper Adversarial Reprogramming of Neural Networks - Ian Goodfellow et. al. - ICLR 2019 </br>
Done as part of CS663 Digital Image Processing course </br>
</br>
Parth Shettiwar </br>
Yash Jain </br>
Devansh Garg </br>


## 1 Introduction


The aim is to implement an adversarial attack on a neural network so that it produces different outputs which
might be favourable to the attacker. Unlike other attacks where the aim is to degrade the performance of the
model, here we will reprogram the model to perform a specific task. The main difference between this method
and Transfer learning is that the weights are changed(trained) in the latter. Here we will just modify the input to
the neural network model (trained on a particular task) with weights unchanged, such that it can be used to
perform a different task. For example, Inception V3 model trained on ImageNet dataset, can be reprogrammed
to classify MNIST digits or count the number of squares in an image. In practice, there is no constraint that
adversarial attacks should adhere to some framework. Thus, it is crucial to proactively anticipate other unexplored
adversarial goals in order to make machine learning systems more secure

## 2 Methods


The overall task is to reprogram the model to perform a task chosen by the attacker, without the attacker needing to compute the specific desired output. Suppose the model takes inputs x and produces output f (x) originally. We would like to perform an adversarial task for inputs x' and producing g (x') as outputs. To achieve the reprogramming,we would learn the adversarial reprogramming functions hf and hg such that hf transform the input x' to the domain of x which can be fed to f (x). hg would convert the outputs back to the domain of g (x').
Effectively


**hg ( f ( hf (x'))) = g (x').**


where

- _x_ is a small image
- _g_ is a function that processes small images
- _hf_ is a function such that puts it the small image _x_ at the centre of some larger image _x_ which is to be learned
- _f_ is a function that processes these larger images(in our case it is the Resnet50 model) to classify them as one of ImageNet labels
- _hg_ is just the hard coded mapping between the produced labels and the adversarial labels

The following example (Figure 1) illustrates this:
The adversarial task is to count squares in an image. For that we must map the ImageNet labels, say first 10, to adversarial task labels. Then we take the adversarial image and put that in a centre of a larger image which has its corresponding centre portion removed out. The larger image is to be learned here. We then pass this new image to the ImageNet classifier(say Resnet or Inception model), which will classify it as one of ImageNet labels.
Finally the hardcoded mapping function _hg_ will classify it as one of the adversarial labels.
We have implemented this with MNIST classification as the adverserial task.


Note: This method is different from Transfer learning where we use weights trained on one dataset and train the top layers to use it for classification of some other dataset. Here we don’t change the weights of the Resnet model, rather we are modifying the input to the model.

![](https://github.com/jinga-lala/Adversarial-Reprogramming-of-Neural-Networks/blob/master/Figure_3.png)
Figure1: Reprogramming the neural network

## 2.1 Implementation Detail


The adversarial program - say _W_ is to be learned of size _R^n_ × _n_ × 3 where _n_ is the ImageNet image width This
program will be same for all images and not specific to a single image. Then we apply a mask so that we can
accommodate the adversarial data.

**_P_ = _tanh_ ( _W &#183; M_ )**


The mask M is such that it is all 1 except the central portion where it is 0.Also we use tanh function as it keeps
the output between (-1,1), which is required for the ImageNet as input.Also we keep the adversarial data at centre
but some other scheme is also possible.Intuitively we should keep it symmetric, hence its kept at centre.Next we
add the adversarial image( _X'_), on which we apply the adversarial task, to the computed quantity _P_.


**Xadv = X'+ P**

The image _X'_ will go in the Central portion of P where it is masked off.
Let _P_ ( _y_ | _X_ ) be the probability that an ImageNet classifier gives to ImageNet label y &epsilon; 1,... , 1000, given
an input image X. The adversarial goal is thus to maximize the probability _P_ ( _hg_ ( _yadv_ )| _Xadv_ ).The optimization
problem is

**_W_ = arg min <sub>_W_</sub>(− _logP_ ( _hg_ ( _yadv_ )| _Xadv_ ) + _λ_ || _W_ ||<sup>2</sup><sub>_F_</sub>**

where _λ_ is a regulariser to avoid overfitting and function _hg_ is a hardcoded mapping between ImageNet labels and
adversarial task labels.The cost of this computation is minimal and attacker needs only to store the program and
add it to the data, leaving the majority of computation to the target network

## 2.2 Hyperparameters

- Learning rate = 0.05 with Adam optimizer
- Decay = 0.
- Steps per epoch = 2
- _λ_ = 1e-8
- epochs = 10


## 2.3 Observation and Results

Our implementation is on MNIST digits database(acting as adversarial database) acting on a ImageNet classifier.
We took 60,000 MNIST training images and 10,000 test images.Further, We have used Resnet 50 v2 as the
ImageNet classifier. The first 10 ImageNet labels were assigned to the MNIST digits. The adversarial program
looks like this :
MNIST digit 1         |  MNIST digit 9
:-------------------------:|:-------------------------:
![](https://github.com/jinga-lala/Adversarial-Reprogramming-of-Neural-Networks/blob/master/Figure_1.png) | ![](https://github.com/jinga-lala/Adversarial-Reprogramming-of-Neural-Networks/blob/master/Figure_2.png)

Figure 2: Adversarial programme to classify MNIST digits using ImageNet classifier. Left: MNIST digit 1, Right: MNIST digit 9 embedded in a adverserial larger image


Here the image is of size 224 x 224 x 3 of which 28 x 28 x 3 centre block is occupied by MNIST image.
For digit 1, we accurately classified it, however digit 9 was classfied as 7.
About 79.75 % accuracy with L2-cross entropy loss of 0.78 was achieved on MNIST Database

## 2.4 Conclusion and Future Use

We saw adversarial reprogramming on classification tasks in the image domain.It was seen that trained networks can be reprogrammed to classify MNIST examples, which do not have any resemblance to images.
A possible future use could be in RNN (Recurrent Neural Networks), where if find suitable inputs to RNN network, it could be able to perform number of simple operations like increment counter, decrement counter etc.

## 2.5 References

```
Adversarial Reprogramming of Neural Networks - Ian Goodfellow, Gamaleldin F.Elsayed - ICLR 2019
```


