import os
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical


# Function to create a directory to save the MNIST images
def create_mnist_image_directory(directory):
    os.makedirs(directory, exist_ok=True)


# Function to save MNIST images in a directory
def save_mnist_images(directory):
    (_, _), (X_test, y_test) = mnist.load_data()
    for i in range(len(X_test)):
        img_path = os.path.join(directory, f"{i}.png")
        cv2.imwrite(img_path, X_test[i])


# Function to load MNIST images from a directory and preprocess for training
def load_mnist_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)

            label = ''.join(filter(str.isdigit, filename))  # Extract only digits from the filename
            label = int(label) % 10  # Ensure labels are within 0-9 range
            labels.append(label)
    return np.array(images), np.array(labels)


# Function to build and train the model
def train_model_on_mnist_images(images_dir, optimizer_choice, num_epochs):
    X_train, y_train = load_mnist_images(images_dir)
    y_train = to_categorical(y_train, num_classes=10)

    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    optimizer = None
    if optimizer_choice == '1':
        optimizer = 'adam'
        print("Adam optimizer: Combines the advantages of AdaGrad and RMSProp. "
              "It adapts learning rates individually for each parameter.")
    elif optimizer_choice == '2':
        optimizer = 'rmsprop'
        print("RMSProp optimizer: Divides the learning rate by an exponentially decaying average of squared gradients. "
              "It helps to alleviate the diminishing learning rates of AdaGrad.")
    elif optimizer_choice == '3':
        optimizer = 'sgd'
        print("SGD (Stochastic Gradient Descent) optimizer: Basic gradient descent with momentum. "
              "It updates the weights in the direction of the negative gradient of the loss function.")
    else:
        print("Invalid optimizer choice. Using default optimizer 'adam'.")
        optimizer = 'adam'

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=32)

    # Save the trained model
    model.save('mnist_model.h5')  # Saving the model in HDF5 format

    return model


# Function to predict on an image
def predict_image(model, image):
    try:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error loading the image. Please check the path.")
            return None

        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        return predicted_class
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function to add adjustable noise to an image and predict
def predict_noisy_image(model, image, noise_level=0.5):
    try:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Error loading the image. Please check the path.")
            return None, None

        # Add adjustable noise and normalize
        noisy_img = img + np.random.normal(0, noise_level, img.shape)
        noisy_img = np.clip(noisy_img, 0, 255)
        noisy_img = noisy_img / 255.0

        # Resize noisy image to match model input shape
        noisy_img = cv2.resize(noisy_img, (28, 28))
        noisy_img = noisy_img.reshape(28, 28, 1)  # Reshape to (28, 28, 1) for saving

        # Save the noisy image
        noisy_image_path = 'noisy_image.png'
        cv2.imwrite(noisy_image_path, (noisy_img * 255).astype(np.uint8))

        # Make prediction on the noisy image
        noisy_img = noisy_img.reshape(1, 28, 28, 1)  # Reshape back to (1, 28, 28, 1) for prediction
        prediction = model.predict(noisy_img)
        predicted_class = np.argmax(prediction)
        return predicted_class, noisy_image_path
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def main():
    mnist_images_dir = 'mnist_images'

    create_mnist_image_directory(mnist_images_dir)
    save_mnist_images(mnist_images_dir)

    print("MNIST images downloaded!")

    optimizer_choice = input("Choose an optimizer:\n"
                             "1. Adam\n"
                             "2. RMSProp\n"
                             "3. SGD\n"
                             "Enter your choice (1/2/3): ")

    while True:
        try:
            num_epochs = int(input("Enter the number of epochs: "))
            if num_epochs > 0:  # Ensure the provided value is valid (positive integer)
                print(f"Number of epochs: {num_epochs}")  # Check if the input is correctly captured
                break  # Break the loop if input is successfully converted to a positive integer
            else:
                print("Please enter a positive integer for the number of epochs.")
        except ValueError:
            print("Please enter a valid integer for the number of epochs.")
    print("Reached after entering the number of epochs.")  # Add a print statement to check program flow

    model = train_model_on_mnist_images(mnist_images_dir, optimizer_choice, num_epochs)
    print("Model trained on downloaded MNIST images!")

    print("Before the while loop for choices")  # Add a print statement to track program flow

    while True:
        print("Inside the while loop for choices")  # Add a print statement to track program flow
        print("\nChoose an option:")
        print("1. Predict on an image")
        print("2. Predict on a noisy image")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            image_path = input("Enter the path of the image: ")
            prediction = predict_image(model, image_path)
            print(f"Predicted number: {prediction}")

        elif choice == '2':
            image_path = input("Enter the path of the image: ")
            noise_level = float(input("Enter the noise level (e.g., 0.5): "))
            prediction, noisy_image_path = predict_noisy_image(model, image_path, noise_level)
            if prediction is not None and noisy_image_path is not None:
                print(f"Predicted number from noisy image: {prediction}")
                print(f"Noisy image saved at: {noisy_image_path}")

        elif choice == '3':
            print("Exiting the program...")
            break

        else:
            print("Invalid choice. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
