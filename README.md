This repository contains a Python-based  Face Recognition project a type of machine learning or deep learning model designed to identify or verify a person's identity by analyzing and comparing facial features in images or videos. These models are widely used in applications like security systems, user authentication, social media tagging, and more.

** Objective:

Develop a secure and efficient face recognition system.

Implement a user-friendly interface for data capture, training, and recognition.

Integrate the system into a specific use case, such as access control, attendance monitoring, or user authentication.
 ** Features:

1. Real-Time Face Detection and Recognition:
Ability to process live video feeds or images and identify faces within milliseconds.


2. Database Management:
Maintain a database of known faces and corresponding IDs for recognition.


3. Scalability:
Handle large datasets of faces for enterprise-level applications.


4. Performance Metrics:
Evaluate accuracy, precision, recall, and speed to optimize the system for high performance.

** Programming Languages: Python

** Libraries and Frameworks:
- Python 3.7 or higher
- OpenCV library
- NumPy library
- A pre-trained face model from Google Teachable Machine


** Installlation of facefinder+ 
>> Step1: Open chrome and search for Google Teachable Machines website(https://teachablemachine.withgoogle.com/) and click on Get started.
>> ![1000165251](https://github.com/user-attachments/assets/98cd2596-1687-4c63-95c2-ddc1a96c3d0c)
>> ![1000165220](https://github.com/user-attachments/assets/3c0932b0-2312-4839-91c5-62e1a47de41f)


 


>> Step2: And under the New Project section, select the Image Model folder.
>> ![1000165221](https://github.com/user-attachments/assets/0c9d15c5-66de-40d9-bcd7-ea34e9a0f7e6)
>> import face_recognition

# Load the image file into a NumPy array
image = face_recognition.load_image_file("your_file.jpg")




>> Step3: Select the Standard Image Model option.![1000165246](https://github.com/user-attachments/assets/d9fde2a6-6cb4-4693-b61d-87a7bd078b95)
>># Find all face landmarks in the image
face_landmarks_list = face_recognition.face_landmarks(image)
Description:-face_landmarks() identifies facial features like eyes, nose, mouth, eyebrows, and chin and returns a list of dictionaries. Each dictionary corresponds to a detected face and contains keys (facial features) and their associated points.

>> Step4: Determine the Required classes and upload the photos using webcam and google drive, And Train the Model. You could see the comparision percentage for the given classes.![1000165248](https://github.com/user-attachments/assets/df59bade-6fa8-456a-af64-19f6b41b4e99)



>> Step5: Click on Export the Model.![1000165247](https://github.com/user-attachments/assets/e40b99ed-09e5-41a3-8187-def81ffeebad)


>> Step6: Under the tensorflow tab, select Download the model option. And a .zip file would be downloaded. 


>> Step8: To run it in local system, paste the Given code in pycharm or any other code Interpreter.

>> Step9: Open the .zip file and copy the .h5 file and .txt file. And the save the files to the project path to Run the code.

>> Step10: Open the terminal and install the required libraries for the project.
Tip-- chech the right python versions and tensorflow versions to avoid the syntax errors.

>> Step11: Run the code and, Hence the facefinder+ works.
