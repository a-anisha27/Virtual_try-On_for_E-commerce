import cv2
from google.colab.patches import cv2_imshow
from google.colab import files
import numpy as np
import dlib
import io

# Upload image
uploaded = files.upload()
file_name = next(iter(uploaded))
image = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), cv2.IMREAD_COLOR)

# Upload glasses and shape predictor
print("Upload 'glasses.png' and 'shape_predictor_68_face_landmarks.dat'")
uploaded_files = files.upload()

# Load dlib's face detector and shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load glasses image with alpha channel
glasses_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

def overlay_glasses(frame, glasses, landmarks):
    left_eye = landmarks[36]
    right_eye = landmarks[45]

    glasses_width = int(abs(right_eye[0] - left_eye[0]) * 2)
    glasses_height = int(glasses_width * (glasses.shape[0] / glasses.shape[1]))

    top_left = (int(left_eye[0] - glasses_width / 4), int(left_eye[1] - glasses_height / 2))

    resized_glasses = cv2.resize(glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

    for i in range(resized_glasses.shape[0]):
        for j in range(resized_glasses.shape[1]):
            y = top_left[1] + i
            x = top_left[0] + j

            if y >= frame.shape[0] or x >= frame.shape[1] or x < 0 or y < 0:
                continue

            alpha = resized_glasses[i, j, 3] / 255.0
            if alpha > 0:
                frame[y, x] = (1 - alpha) * frame[y, x] + alpha * resized_glasses[i, j, :3]

    return frame

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

for face in faces:
    landmarks = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    image = overlay_glasses(image, glasses_img, landmarks)

# Show the final image
cv2_imshow(image)
