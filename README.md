[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project
### Writeup by Ayo Adedeji
---
#### Instructions on how to implement segmentation network, collection training data, train model, and test in simulation are found here [here](./project_setup.md).

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
---
### Achieving Semantic Segmentation using a Fully Convolutional Neural Network 
In a Fully Convolutional Neural Network (FCN), each layer is a convolutional layer. This differs from approaches to classification in typical convolutional network schemes where a fully connected layer (each neuron is connected to each neuron in an antecedent layer allowing for spatial and global integration of multiple features) or a multilayered perceptron is applied at end of a convolutional neural network. A convolutional layer is applied at the end of FCNs to classify (i.e label with a color) pixel state of an input image. Semantic segmentation is achieved by producing an output layer with spatially matched dimensionality in which each pixel from the input image is classified and spatial information from input layer ultimately retained — this faciliates identification of objects through encoded location in local space and is the basis of training an FCN to get a simulated quadcopter to follow a target in space. FCN architecture for semantic segmentation consists of an encoder network (a series of convolutional layers that reduces input layer to a 1x1 convolution layer) followed by a decoder network (a series of convolutional layers that projects the finer, higher resolution features of the encoder layers into the features of the output layer with input layer spatially matched dimensionality).  

#### Encoder Network
Each layer of the encoder network is a separable convolution layer that reduces the number of parameters as would be required by a regular convolution layer. This reduction of needed parameters ultimately functions to improve runtime efficiency and also reduce overfitting by providing less parameters to which to fit to (obliges network to focus more on generalized patterns)
* Separable convolution layers for each encoder block in the FCN was generated through the following helper function
* A ReLU activation function, *same* padding parameter, and kernel_size = 3 are applied by default.
```python
def encoder_block(input_layer, filters, strides):
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
    
 def separable_conv2d_batchnorm(input_layer, filters, strides=1):
  output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                           padding='same', activation='relu')(input_layer)

  output_layer = layers.BatchNormalization()(output_layer) 
  return output_layer
```
#### Decoder Network
To project the lower resolution features of the encoder layers into the higher resolution features of each output layer (in other words, to achieve upsampling), I used the bilinearly upsampling approach of averaging the four nearest pixels located diagonally to each pixel to arrive at a new pixel value for new pixel values.
* Upsampling by a factor of 2 was implemented through the following helper function (with a keras backend)
```python
def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer
   
def decoder_block(small_ip_layer, large_ip_layer, filters):
    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled_input_layer = bilinear_upsample(input_layer=small_ip_layer)
    # Concatenate the upsampled and large input layers using layers.concatenate
    concatenated_upsampled_and_large_input_layer = layers.concatenate([upsampled_input_layer, large_ip_layer])
    # Add some number of separable convolution layers
    separable_convolution_layer_1 = separable_conv2d_batchnorm(input_layer=concatenated_upsampled_and_large_input_layer, filters=filters)
    separable_convolution_layer_2 = separable_conv2d_batchnorm(input_layer=separable_convolution_layer_1, filters=filters)
    output_layer = separable_conv2d_batchnorm(input_layer=separable_convolution_layer_2, filters=filters)
    return output_layer
```
Layer concatenation and three following separable convolution layers are implemented after upsampling small input layer to In to optimize the preservation of fine, higher resolution details from preceding layers as the decoder network decodes and upsamples to a size that has equivalent dimensionality with initial input layer. Concatenating layers is equivalent to element-wise addition of layers but does not require all layers to have same to execute operation.

##### Batch Normalization
Each layer within the encoder and decoder blocks is normalized. Normalizing inputs to a layer enhances performance because input data with more variance around mean will result in opinionated weighting that harshly penalizes increasing distance from central mean peak; input data with less variance around mean results in less opinionated weighting at start that only becomes more opinionated with training / learning of patterns within data. Batch normalization encompasses treating each layer as input layer to a smaller network and requires normalization of each layer's inputs.
```python
output_layer = layers.BatchNormalization()(output_layer) 
```
### FCN Model

```python
def fcn_model(inputs, num_classes):
    # Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoder_block_layer_1 = encoder_block(input_layer=inputs, filters=32, strides=2)
    encoder_block_layer_2 = encoder_block(input_layer=encoder_block_layer_1, filters=64, strides=2)
    encoder_block_layer_3 = encoder_block(input_layer=encoder_block_layer_2, filters=128, strides=2)
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    one_to_one_convolution_layer = conv2d_batchnorm(input_layer=encoder_block_layer_3, filters=256, kernel_size=1, strides=1)
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_block_layer_4 = decoder_block(small_ip_layer=one_to_one_convolution_layer, large_ip_layer=encoder_block_layer_2, filters=128)
    decoder_block_layer_5 = decoder_block(small_ip_layer=decoder_block_layer_4, large_ip_layer=encoder_block_layer_1, filters=64)
    x = decoder_block(small_ip_layer=decoder_block_layer_5, large_ip_layer=inputs, filters=32)
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)
```
The architecture of the FCN model of my project consists of three encoder layers and three decoder block layers. After each encoder layer, the depth of the model doubles. I choose to have the depth double after each layer in relation to value of the stride parameter choosen for each encoder layer of 2. Moreover, given that the width-height dimensionality of each encoder layer reduces by a factor of ~2 after each convolution given stride of 2, to prevent loss of finer details and to optimize preservation of spatial information, I translated reduction in width-height dimensionality after each convolution to increased depth / more filters are present after each convolution.
* Regarding the 1 x 1 convolution layer, in general, a 1 x 1 convolution is a feature pooling technique that is implemented to reduce filter depth which in turn optimizes for computational load amongst other things. I found that when I reduced the filter depth of the 1 x 1 convolution layer performance of the FNC model suffered. I suppose this reduction in performance stems from a loss of higher detail resolution due to reductive feature pooling. I found that keeping filter depth at 128 or further doubling it to 256 optimized for FCN model performance. As a side note, I tested out a filter depth of 256 to see how that would affect performance and it seemed to give a slight boost in performance to the FCN model — I suppose this is due to a degree of enhanced fitting from expanded features.


##### Compute Color Histograms
* In process of reading RGB data from the point clouds from each snapshot, I converted RGB data to HSV (hue-saturation-value) color space to increase robustness of object recognition (RGB is sensitive to changes in brightness etc.)
* I used 32 bins, so roughly 12.5% of color data would fall into each bin (initially, I experimented with larger number of bins but didn't see increase in performance enough to justify increase in processing time larger number bins brought on)

[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Setup Instructions
**Clone the repository**
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running `preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run `preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.
Rename or move `data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and  `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate your new training and validation sets. 


## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py my_amazing_model.h5
```

**Note:** If you'd like to see an overlay of the detected region on each camera frame from the drone, simply pass the `--pred_viz` parameter to `follower.py`
