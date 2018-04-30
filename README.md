# Distributed Deep Learning with TensorFlow
## Getting Started 
This course is aimed at intermediate machine learning engineers, DevOps, technology architects and programmers who are interested in knowing more about deep learning, especially distributed deep learning, TensorFlow, Google Cloud and Keras. We are here to give you the skills to analyze large volumes of data in a distributed way for a production level system. After the course, you will be able to have a solid background in how to scale-out machine learning algorithms in general and deep learning in particular. 

We have designed the course to provide you with the right blend of hands-on, theory and best practices in this rapidly developing area while providing grounding in essential concepts which remain timeless.

 Tools and frameworks like, `Keras`, `TensorFlow`, and `Google Cloud` are used to showcase the strengths of various approaches, trade-offs and building blocks for creating real-world examples of distributed deep learning models.


### Prerequisites
This course is for intermediate machine learners like you who want to learn more about deep learning, how to scale out your deep learning model, and then quickly turn around and use the tools and techniques you are about to learn from this course to solve your tricky deep learning tasks. 

You will be successful in this course if you have a basic knowledge of computer programming especially Python programming language. Also some familiarity with deep learning like neural networks will be helpful. 

In this course, you will need a Google Cloud free tier account. Note that you won't be charged by creating the account. Instead, you can get `$300` credit to spend on Google Cloud Platform for 12 months and access to the Always Free tier to try participating products at no charge. By going through this course, you will probably need to spend at most `$50` out of your `$300` free credit. 

### Built with 
* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/) 
* [Google Cloud MLE](https://cloud.google.com/ml-engine/)

### Versioning 
* [Keras](https://github.com/keras-team/keras) `2.1.6`
* [TensorFlow](https://github.com/tensorflow/tensorflow) `1.8`
* [Google Cloud MLE](https://cloud.google.com/source-repositories/) `latest`

### Installing 
1. Keras
```
sudo pip install keras
```
2. TensorFlow 
```
sudo pip install tensorflow-gpu
```
**OR**
```
sudo pip install tensorflow
```
3. Google Cloud MLE 
https://cloud.google.com/sdk/
> Installation details will be explained in [Section III](https://github.com/christianramsey/Tensorflow-for-Distributed-Deep-Learning)

### Authors 
**Christian Fanli Ramsey** 
* LinkedIn : https://www.linkedin.com/in/christianramsey/


* Tumblr : https://www.tumblr.com/blog/anthrochristianramsey

**Haohan Wang** 
* LinkedIn : https://www.linkedin.com/in/haohanw/


* Tumblr : https://www.tumblr.com/blog/haohanwang 

***
### Content
**Installation and Setup - [Setup Distributed Deep Learning Enviornment](https://github.com/mxmnml/Distributed-Deep-Learning-with-Tensorflow/tree/master/0.%20Setup%20distributed%20deep%20learning%20enviornment)**
* Nvidia Setup
* Anaconda Setup
* Nvidia and Docker
* Requirements

**SECTION I – [Deep Learning with Keras](https://github.com/mxmnml/Distributed-Deep-Learning-with-Tensorflow/tree/master/1.%20Deep%20Learning%20with%20Keras)**
* 1.1 Keras Introduction
* 1.2 Review of backends Theano, Tensorflow, and Mxnet
* 1.3 Design and compile a model
* 1.4 Keras Model Training, Evaluation and Prediction
* 1.5 Training with augmentation 
* 1.6 Training Image data on the disk with Transfer Learning and Data augmentation 
-----

**SECTION II – [Scaling Deep Learning using Keras and Tensorflow](https://github.com/mxmnml/Distributed-Deep-Learning-with-Tensorflow/tree/master/2.%20Distributed%20TensorFlow%20%26%20Keras)**
* 2.1 Tensorflow Introduction
* 2.2 Tensorboard Introduction
* 2.3 Types of Parallelism in Deep Learning – Synchronous vs Asynchronous
* 2.4 Distributed Deep Learning with tensorflow 
* 2.5 Configuring Keras to use tensorflow for Distributed problems 
---

**SECTION III - [Distirbuted Deep Learning with Google Cloud MLE](https://github.com/mxmnml/Distributed-Deep-Learning-with-Tensorflow/tree/master/3.%20Distributed%20Deep%20Learning%20with%20Google%20ML%20Engine)**
* 3.1 Representing data in TensorFlow
* 3.2 Dive into Estimators
* 3.3 Creating your Data Input Pipeline
* 3.4 Creating your Estimator
* 3.5 Packaging your model/trajectory 
* 3.6 Training in the Cloud
* 3.7 Hyperparameters Tuning
* 3.75 Automated Hyperparameter Tuning/ Trajectory 
* 3.8 Cloud Prediction/Trajectory/rnn_training_files
* 3.9 Deploying your Model to the Cloud to Prediction 
* 3.10 Creating your Machine Learning API

##### More Links for Section III: 
* DDL- MLE: https://github.com/christianramsey/ddl_mle
* GCMLE: https://github.com/christianramsey/gcmle.git
* Reading Data: https://github.com/christianramsey/reading-data.git
* Intro to Estimator: https://github.com/christianramsey/intro-to-estimators.git
* Read with Beam: https://github.com/christianramsey/read_with_beam.git
* Hyperparameter: https://github.com/christianramsey/GCMLE---PHASE-II-Hyperparameters-.git
* DDL: https://github.com/christianramsey/ddl.git
* Export DataFlow: https://github.com/christianramsey/dataflow_export_gcmle.git
* Streaming: https://github.com/christianramsey/streaming_gcmle.git
