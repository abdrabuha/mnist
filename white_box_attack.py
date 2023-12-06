import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load your trained model
saved_model = load_model('mnist_model.h5')


# Function to generate adversarial example using FGSM
def generate_adversarial_example_FGSM(model, input_image, epsilon):
    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)

    gradient = tape.gradient(prediction, input_tensor)
    perturbation = epsilon * tf.sign(gradient)
    perturbation = tf.clip_by_value(perturbation, -0.1, 0.1)

    adversarial_example = input_tensor + perturbation
    adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)

    return adversarial_example.numpy()


# Function to generate adversarial example using PGD
def generate_adversarial_example_PGD(model, input_image, true_label, epsilon=0.1, alpha=0.01, iterations=40):
    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
    adversarial_example = tf.identity(input_tensor)

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_example)
            prediction = model(adversarial_example)
            loss = -tf.keras.losses.CategoricalCrossentropy()(true_label, prediction)

        gradient = tape.gradient(loss, adversarial_example)
        perturbation = alpha * tf.sign(gradient)
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)

        adversarial_example = tf.clip_by_value(adversarial_example + perturbation, 0, 1)

    return adversarial_example.numpy()


# Load the test image
test_image_path = '2.png'  # Replace with your test image path
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (28, 28))
test_image = test_image / 255.0
test_image = test_image.reshape(1, 28, 28, 1)

# Generate a true label for the test image
num_classes = 10
true_class = 3  # Example class (change this according to your data)
true_label = np.zeros(num_classes)
true_label[true_class] = 1

# Get user choice for adversarial attack method
attack_choice = input("Choose adversarial attack method (FGSM/PGD): ")

if attack_choice.upper() == 'FGSM':
    epsilon = float(input("Enter epsilon value for FGSM attack: "))

    # Generate adversarial example using FGSM
    adversarial_example = generate_adversarial_example_FGSM(saved_model, test_image, epsilon)
    adversarial_example_image = Image.fromarray((adversarial_example[0] * 255).astype(np.uint8).reshape(28, 28))

    # Save the adversarial example image
    adversarial_example_image.save('adversarial_example_FGSM.png')  # Replace with desired filename and format

    # Calculate the difference between the original and adversarial images
    difference = np.abs(
        (adversarial_example[0] * 255).astype(np.uint8).reshape(28, 28) - test_image.reshape(28, 28) * 255)

    # Create a heatmap by amplifying the differences for visualization
    heatmap = plt.imshow(difference, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.axis('off')

    # Save the heatmap as an image
    plt.savefig('adversarial_heatmap_FGSM.png')  # Replace with desired filename and format

    # Test the model's prediction on original and adversarial examples
    original_prediction = np.argmax(saved_model.predict(test_image))
    adversarial_prediction = np.argmax(saved_model.predict(adversarial_example))

    print(f"Original Prediction: {original_prediction}")
    print(f"Adversarial Prediction (FGSM): {adversarial_prediction}")

elif attack_choice.upper() == 'PGD':
    epsilon = float(input("Enter epsilon value for PGD attack: "))
    alpha = float(input("Enter alpha value for PGD attack: "))
    iterations = int(input("Enter the number of iterations for PGD attack: "))

    # Generate adversarial example using PGD
    adversarial_example = generate_adversarial_example_PGD(saved_model, test_image, true_label, epsilon=epsilon,
                                                           alpha=alpha, iterations=iterations)

    # Calculate the difference between the original and adversarial images
    difference = np.abs(
        (adversarial_example[0] * 255).astype(np.uint8).reshape(28, 28) - test_image.reshape(28, 28) * 255)

    # Create a heatmap by amplifying the differences for visualization
    heatmap = plt.imshow(difference, cmap='hot', interpolation='nearest')
    plt.colorbar(heatmap)
    plt.axis('off')

    # Save the heatmap as an image
    plt.savefig('adversarial_heatmap_PGD.png')  # Replace with desired filename and format

    # Test the model's prediction on original and adversarial examples
    original_prediction = np.argmax(saved_model.predict(test_image))
    adversarial_prediction = np.argmax(saved_model.predict(adversarial_example))

    print(f"Original Prediction: {original_prediction}")
    print(f"Adversarial Prediction (PGD): {adversarial_prediction}")

else:
    print("Invalid choice. Please choose either 'FGSM' or 'PGD'.")
