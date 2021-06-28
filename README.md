# Deep Learning
Custom deep learing model for Gachon Deep Learning Class Competition (2020.04 ~ 2020.05)   
Gachon Deep Learning Class Competition : https://www.kaggle.com/c/gachon-deeplearning   
   
Created by __Team I__ (Lee NamJun, Yeo JunKu, Cho SoYeong, Choi JiWon)

# Dataset Dataset Description
We will use a subset of Yoga-82 dataset: https://sites.google.com/view/yoga-82/   
_balancing, inverted, reclining, sitting, standing, and wheel._

# Goal of Competition
Through this competition,
we hope to improve our deep learning and PyTorch coding skills so that we can finally have a wide perspective on the deep learning model.

# Our Approach
The primary logic of our approach is find out the best combination of element of model (hyper-parameters : optimizers, batch size, ...) with less epoch, then train model with the combination with more epoch.    
So, we use __Tuner__ to find out the best combination of hyper parameters and optimizer.
   
   
We used the following parameters:
~~~
config = {
    "lr": [0.001, 0.005, 0.0005],
    "batch_size": [64, 32, 16, 8],
    "optimizer": ['SGD','Adagrad', 'Adam']
}
~~~

As a result, we increased the accuracy from 71% to 85% using Tuner.
