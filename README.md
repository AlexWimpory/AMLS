# AMLS Coursework
This project uses neural networks to perform 4 classification tasks.

Directory Structure:
* A1: Gender detection: male or female
* A2: Emotion detection: smiling or not smiling
* B1: Face shape recognition: 5 types of face shapes
* B2: Eye color recognition: 5 different eye colors
* Datasets: Celeba, Celeba_test and Cartoon_set, Cartoon_set_test

Each task is self-sufficient but the code was designed to be reusable
 therefore there is a lot of duplicate code in each directory.

Packages required found in requirements.txt and can be installed using pip install -r requirements.txt.
  However I did have some issues getting dlib to work on my Windows machine, but this should not matter as
  explained in the report, this did not end up being used.
  
  
Requirements.txt

* pandas==1.1.4\
Used to store features and labels in a data frame which provides an easy input for models
 and allows for analysis of the input data.
* tensorflow==2.3.1\
Using keras to build neural networks which is a deep learning application programming interface (API), built on top of TensorFlow.
In particular, the sequential model API is used where the network is defined layer by layer.
  The added flexibility of the functional model API is not needed which allows the connections between each layer to be specified.
* matplotlib==3.3.2\
A library for plotting data in python.
* opencv-python==4.4.0.46\
A library which provides tools for computer vision problems, which is relevant in 
this case as pictures have to be converted into useable features for the models.
* numpy~=1.18.5\
Arrays are not traditionally needed for Python as there are lists and dictionaries. 
 When there are required NumPy provides tools for this.
* scikit-learn==0.23.2\
A machine learning library which features easy implementation of standard regression and classification methodologies.
* seaborn==0.11.0\
A library for plotting data in python which is used for visualizing the confusion matrices.
* dlib==19.21.1\
A toolkit for real-world machine learning problems which is used for frontal face detection 
(although this is not required as described later in the report this has no effect on the results).

Main.py

* First select the task (A1, A2, B1, B2) and then the operation (Train, Evaluate, Predict).
* Train will preprocess the data in the directory in which the training set should always be stored (celeba or cartoon_set).  It will then save the data (a lot quicker to load the file when experimenting with the code) before training and saving the model.
* Evaluate will produce an accuracy based on the images in the _test directory and the associated labels.
* Predict will run a single image through the model and produce a predicted class and the associated probabilities
