# Implementing-a-Deep-Convolutional-Autoencoder-for-Noise-Removal-on-MNIST-Images

# 1.Project title and description:

Written based on TensorFlow 2.x and the Keras library for creating and training deep learning models for implementing autoencoders for image reconstruction tasks. Implementation of a deep convolutional autoencoder for noise removal on MNIST images.


# 2.MNIST Handwritten Digits Dataset

MNIST (Modified National Institute of Standards and Technology) is a classical dataset widely used in computer vision and machine learning research. The dataset consists of handwritten digit images, each representing a single digit between 0 and 9. The following is a detailed description of the MNIST dataset:

Data size: the MNIST dataset contains 60,000 training samples and 10,000 test samples. Each sample is a grayscale image of 28x28 pixels.

Classification task: The MNIST dataset is typically used for image classification tasks. Each sample has a numeric label associated with it, indicating the handwritten digits displayed in the image.

Categories: The MNIST dataset contains 10 categories corresponding to the digits 0 to 9.

Data Distribution: The images in the dataset are handwritten from different authors and therefore contain a wide range of different handwriting styles and handwriting.

Application areas: The MNIST dataset is widely used for testing and evaluating various machine learning algorithms, especially deep learning models for image classification and digit recognition.

Data Preprocessing: Typically, MNIST images need to be normalized to scale the pixel values to the range of [0, 1] for better learning by the model. In addition, noise reduction or data enhancement operations are sometimes performed on the images to improve the performance of the model.

# 3.Running environment:

Google Colab (T4GPU)

Running time: about five minutes 

# 4.Parameter settings

noise_factor = 0.7

epochs=50

batch_size=128

# 5.autoencoder

Autoencoder is an unsupervised learning algorithm that is commonly used for tasks such as dimensionality reduction, feature learning and image denoising. Its main idea is to try to learn how to recode the input data into itself so that useful features of the data can be captured and the dimensionality of the data can be reduced.Autoencoder usually consists of two parts: an Encoder and a Decoder.

The following are the basic principles and components of Autoencoder:

Encoder: The Encoder part maps the input data to a low-dimensional representation, often called the encoding or hidden layer. The goal of the encoder is to capture key features of the input data while reducing the dimensionality of the data. This process can be considered as a feature extractor.

Decoder: The decoder part remaps the encoded representation back to the dimensionality of the original input data. The goal of the decoder is to reconstruct the data as close as possible to the original input. This process can be thought of as a generator.

Loss Function: The loss function of Autoencoder is usually the reconstruction error, which is a measure of the difference between the decoder output and the original input. Common loss functions include Mean Squared Error or Binary Cross-Entropy, depending on the type of input data.

The training process of Autoencoder consists of the following steps:

The input data is passed through an encoder to generate a code.

The encoded representation is passed through the decoder to generate the reconstructed data.

The loss of the reconstructed data with respect to the original input is computed.

Parameters of the encoder and decoder are adjusted to minimize the loss using the back propagation algorithm.

Autoencoder can be used for a variety of applications including image denoising, feature learning, data degradation and generating data.


 ![autoencoder](https://github.com/2461011611/Implementing-a-Deep-Convolutional-Autoencoder-for-Noise-Removal-on-MNIST-Images/assets/118686100/fbd5cd2e-82c2-466d-a021-10c5105dc89b)



# 6.Added noise_factor = 0.7

![Added noise_factor = 0 7](https://github.com/2461011611/Implementing-a-Deep-Convolutional-Autoencoder-for-Noise-Removal-on-MNIST-Images/assets/118686100/0e565ae7-0e41-4b4b-8adb-485a869fd2b1)

# 7.Predictions are now made based on the noise data


![Predictions](https://github.com/2461011611/Implementing-a-Deep-Convolutional-Autoencoder-for-Noise-Removal-on-MNIST-Images/assets/118686100/099102c2-cef0-4964-ae68-90e13ea231c1)


