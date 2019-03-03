# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset.png "Dataset"
[real_13]: ./real_images/13.jpg" "13"
[real_14]: ./real_images/14.jpg" "14"
[real_25]: ./real_images/25.jpg" "25"
[real_37]: ./real_images/37.jpg" "37"
[real_38]: ./real_images/38.jpg" "38"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/metrofun/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34779
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how unbalanced is the dataset.

![Dataset][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried normalizing by subtracting dataset's mean and deviding by its variance, seperately per each channel.
Also tried per-image normalization using `per_image_standardization`. However, later switched to using BatchNorm,
so I pruned these steps as redundant.

I have also tried switching to greyscale, interestingly it resulted in a tiny positive impact. In order to make the pipeline as simple as possible, I made a trade off and removed this step as well.

I used albumentation library to augment my dataset to make classes more equally distributed. I used scale +- 10%, rotation +-30 degrees, shift +- 2px.
Empty areas were filled with mirrored image. Additionally I used 20% probability to randomly change contrast and brightnest.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6  	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Batch Norm            |   											|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Batch Norm            |   											|
| RELU					|												|
| Fully connected		| 120        									|
| Batch Norm            |   											|
| RELU					|												|
| Dropout       		|												|
| Fully connected		| 84        									|
| Batch Norm            |   											|
| RELU					|												|
| Dropout       		|												|
| Fully connected		| 43        									|
| Softmax				|             									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used of the shelf Adam with 0.00025 learning rate, 100 epochs and 256 batch size. Plus 0.1 alpha for L2 regularization.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.920
* validation set accuracy of 0.938 (> 0.920 because augmentation introduced more variance into the training data)
* test set accuracy of 0.927

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I started from lecture's LeNet as it was the fastest option and adopted its input and output dimension.
* What were some problems with the initial architecture?
Performance was lower than needed even with augmentation/normalization/grasyscaling etc.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
When training without augmentation, I started overfitting my train data, so I introduces l2 regularization.
To push the accuracy further, I have added batch norm and dropout. Probably I could have made dropout keep rate lower and get rid of L2 all together.
* Which parameters were tuned? How were they adjusted and why?
Doing vanilla parameter tweaking is too time consuming. Alternatively, since I am not proficient with Tensorflow I didn't want to invest time in setting up grid/bayesian search over params, neither looking for impementantation of learning rate finder. So after few restarts I settled for some small learning rate. Plus, when I saw that loss has high varience late in the training, I have doubled the batch size.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I believe tweaking LeNet with extra drects links between convolutional layers and the fully connected later would have made a significant difference. However, I wanted to squeeze 0.93 our of Lenet-5 with minimum number of changes. Otherwise, I would rather try first other known architectures than re-invent it myself.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![13][real_13] ![14][real_14] ![25][real_25] 
![37][real_37] ![38][real_38]

All images were correctly classified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep Right     		| Keep Right   									| 
| Stop       			| Stop   										|
| Yield					| Yield											|
| Road Work      		| Road Work 					 				|
| Go straight or left   | Go straight or left                           |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the end of the notebook.
The model was highly certain when making the predictions for each image: 9.99811113e-01, 9.99972224e-01, 9.99964595e-01, 9.15632546e-01, 1.00000000e+00.
