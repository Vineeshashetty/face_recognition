import cv2  # Install opencv-python
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
try:
    model = load_model("keras_Model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the labels
try:
    with open("labels.txt", "r") as file:
        class_names = [line.strip() for line in file.readlines()]
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()

# Initialize the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame from the webcam
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the image to match model input shape (224x224)
    image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Display the image in a window
    cv2.imshow("Webcam Image", image)

    # Convert image to numpy array, normalize and reshape
    image_array = np.asarray(image, dtype=np.float32)
    image_array = (image_array / 127.5) - 1  # Normalize between [-1,1]
    image_array = image_array.reshape(1, 224, 224, 3)

    # Make prediction
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index] if index < len(class_names) else "Unknown"
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {confidence_score * 100:.2f}%")

    # Exit loop when 'Esc' key (ASCII 27) is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
