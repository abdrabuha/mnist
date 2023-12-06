# mnist
By Abdrabuh Alotaibi

This code is a Python script that performs several operations related to the MNIST dataset using TensorFlow, OpenCV (cv2), and NumPy libraries. Here's a breakdown of what each part of the code does:

    Importing Libraries: Imports necessary libraries including TensorFlow, OpenCV, and NumPy.

    Functions Defined:
        create_mnist_image_directory(directory): Creates a directory to save MNIST images if it doesn't exist.
        save_mnist_images(directory): Saves MNIST images in the specified directory.
        load_mnist_images(directory): Loads MNIST images from the directory, preprocesses them for training, and returns the images and labels.
        train_model_on_mnist_images(images_dir, optimizer_choice, num_epochs): Builds, compiles, and trains a neural network model on MNIST images using the specified optimizer and number of epochs.
        predict_image(model, image): Loads an image, preprocesses it, predicts the digit using the trained model, and returns the predicted class.
        predict_noisy_image(model, image, noise_level=0.5): Adds adjustable noise to an image, predicts the digit using the trained model, and returns the predicted class along with the path of the noisy image saved.

    Main Function (main()):
        Creates a directory for storing MNIST images, saves the MNIST images, and trains the model on these images based on user-selected optimizer and epochs.
        Provides options to predict on a normal image or a noisy image, and allows the user to exit the program.

    User Interaction:
        The script prompts the user to choose an optimizer, enter the number of epochs, and then offers options to perform predictions on images or noisy images.

This code demonstrates a simple pipeline for downloading, processing, training a neural network model on the MNIST dataset, and making predictions on images.
