import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifier for license plates
cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def process_plate(plate):
    # Check if the plate is not emptycccc
    if plate is None or plate.size == 0:
        return None


    # Convert the plate to grayscale
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Apply some morphological operations to clean the image
    kernel = np.ones((3,3), np.uint8)
    plate = cv2.dilate(gray, kernel, iterations=1)
    plate = cv2.erode(plate, kernel, iterations=1)
    
    # Additional processing steps can go here

    return plate

def process_video(filename):
    # Initialize the video capture object
    cap = cv2.VideoCapture(filename)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, None, fx=0.3, fy=0.3)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect license plates in the grayscale frame
        nplate = cascade.detectMultiScale(gray, 1.1, 4)

        # Process each detected license plate
        for (x, y, w, h) in nplate:
            plate = frame[y:y+h, x:x+w]
            processed_plate = process_plate(plate)
            if processed_plate is not None:
                # Draw rectangles around the detected and processed license plates
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def extract_num(filename):
    process_video(filename)

# Call the function with the video file
extract_num("demovideo.mp4")
