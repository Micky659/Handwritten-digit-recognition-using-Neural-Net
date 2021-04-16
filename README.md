# Handwritten-digit-recognition-using-Neural-Net
This program uses neural networks to recognize handwritten numerical digits, there are scripts for both the version of tensorflow. Model uses the traditional mnist  dataset for training and testing, it scored accuracy percentage of 92.06% for tf v1 and 97.01% for tf v2.


## Requirments-
    Python >= 3.6
    Tensorflow
    PILLOW == 8.1.0
    Numpy == 1.19.5
    
    
## Layer architecture-
    Input_layer_size = 784 or (1, 28, 28) for image of dimension 28 x 28
    Hidden_layer1_size = 512  
    Hidden_layer2_size = 256  
    Hidden_layer3_size = 128  
    Output_layer_size = 10 for 0 - 9 digits
 ![](https://github.com/Micky659/Handwritten-digit-recognition-using-Neural-Net/blob/master/figures/Diagram%20of%20neural%20network.png)
    (Image from the book Python Machine Learning Projects
Written by Lisa Tagliaferri, Michelle Morales, Ellie Birbeck, and
Alvin Wan)


## Preprocessed model-
    You can use the preprocessed model created from the script tensorflow v2 with an accuracy of 97 percent to 
    test your own sample images which is stored in the directory preprocessed_model, you can store your test 
    images in the Stock folder and use it in both the scripts.
   
   
## Dataset-
    You don't need to download any dataset and load it extensively just run either of the script and it will 
    automatically download and/or load the dataset and you're good to go.
    

***Feel free to build over my code and use it wisely***
