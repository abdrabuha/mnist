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

`The white_box_attack.py code comprises functionalities to generate adversarial examples using two attack methods, FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent), applied to a pre-trained model on the MNIST dataset. Here's a breakdown of the code:

    Importing Libraries: Imports necessary libraries including NumPy, TensorFlow, OpenCV (cv2), PIL (Image), and Matplotlib.

    Loading the Trained Model: Loads a pre-trained model (mnist_model.h5) that was trained on the MNIST dataset.

    Adversarial Example Generation Functions:
        generate_adversarial_example_FGSM(model, input_image, epsilon): Generates an adversarial example using the FGSM attack method.
        generate_adversarial_example_PGD(model, input_image, true_label, epsilon, alpha, iterations): Generates an adversarial example using the PGD attack method.

    Loading Test Image:
        Loads a test image from the specified path, resizes it to 28x28 (MNIST input shape), and normalizes it.

    User Interaction:
        Prompts the user to choose between FGSM or PGD attack methods.
        For FGSM:
            Requests epsilon value for FGSM attack.
            Generates adversarial example using FGSM.
            Saves the adversarial example image and creates a heatmap visualizing the differences between the original and adversarial images.
            Tests the model's prediction on the original and adversarial examples.
        For PGD:
            Requests epsilon, alpha, and the number of iterations for PGD attack.
            Generates adversarial example using PGD.
            Creates a heatmap visualizing the differences between the original and adversarial images.
            Tests the model's prediction on the original and adversarial examples.

    Output:
        Outputs the original prediction and the model's prediction on the original and adversarial examples.

This code demonstrates the generation of adversarial examples on a pre-trained MNIST model using FGSM and PGD methods and provides visualizations to understand the differences between the original and adversarial images. Additionally, it compares the model predictions on these examples to highlight potential vulnerabilities of the model against adversarial attacks.

Usu it but please dont forget me :)
