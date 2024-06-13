import numpy as np
import face_recognition as fr
import cv2

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load sample pictures and learn how to recognize them
Khalid_image = fr.load_image_file("Khalid.jpg")
Khalid_face_encoding = fr.face_encodings(Khalid_image)[0]

Derhem_image = fr.load_image_file("Derhem.jpg")
Derhem_face_encoding = fr.face_encodings(Derhem_image)[0]

Aiman_image = fr.load_image_file("Aiman.jpg")
Aiman_face_encoding = fr.face_encodings(Aiman_image)[0]

Jurham_image = fr.load_image_file("Jurham.jpg")
Jurham_face_encoding = fr.face_encodings(Jurham_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [Khalid_face_encoding, Derhem_face_encoding, Aiman_face_encoding, Jurham_face_encoding]
known_face_names = ["Khalid", "Derhem", "Aiman", "Jurham"]

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the frame from BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations and face encodings in the current frame
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the known face with the smallest distance to the new face
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
