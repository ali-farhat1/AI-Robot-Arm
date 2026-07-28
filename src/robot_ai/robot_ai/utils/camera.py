import cv2

# Initialize the camera (0 represents the default built-in or first connected USB webcam)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was read correctly, ret will be True
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Display the resulting frame in a window named 'Camera Feed'
    cv2.imshow('Camera Feed', frame)

    # Wait 1 millisecond for any key press. If 'q' is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
