import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX models
palm_sess = ort.InferenceSession("palm_detection_full_inf_post_192x192.onnx")
landmark_sess = ort.InferenceSession("hand_landmark_sparse_Nx3x224x224.onnx")

def preprocess(image, size):
    image = cv2.resize(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0)
    return image

cap = cv2.VideoCapture(0)

palm_input = palm_sess.get_inputs()[0].name
landmark_input = landmark_sess.get_inputs()[0].name

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape

    # Run palm detection
    palm_input_tensor = preprocess(frame, (192, 192))
    palm_output = palm_sess.run(None, {palm_input: palm_input_tensor})[0]

    for box in palm_output:
        score = box[0]
        if score > 0.5:
            cx, cy, w, _ = box[1:5]

            # Expand width and derive height
            w *= 1.4
            aspect_ratio = 1.3
            h = w * aspect_ratio
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h
            cy -= h * 0.15

            # Convert to box bounds
            xmin = int(cx - w / 2)
            xmax = int(cx + w / 2)
            ymin = int(cy - h / 2)
            ymax = int(cy + h / 2)

            # Add top/bottom padding
            ymin = max(0, ymin - 200)
            ymax = min(img_h, ymax + 100)
            pad = 20
            xmin = max(0, xmin - pad)
            xmax = min(img_w, xmax + pad)

            # Draw green box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Crop and run landmark model
            hand_crop = frame[ymin:ymax, xmin:xmax]
            if hand_crop.size == 0:
                continue

            hand_input = preprocess(hand_crop, (224, 224))
            lm_output = landmark_sess.run(None, {landmark_input: hand_input})[0]
            landmarks = lm_output.reshape(21, 3)

            crop_w = xmax - xmin
            crop_h = ymax - ymin
            scale_x = 224 / crop_w
            scale_y = 224 / crop_h

            z_values = [pt[2] for pt in landmarks]
            z_min = min(z_values)
            z_max = max(z_values)

            for x, y, z in landmarks:
                px = int(x / scale_x) + xmin
                py = int(y / scale_y) + ymin

                # Normalize z to range [0, 1]
                z_norm = (z - z_min) / (z_max - z_min + 1e-6)
                r = int(255 * (1 - z_norm))
                g = int(255 * z_norm)
                b = 0

                cv2.circle(frame, (px, py), 5, (b, g, r), -1)

            break

    cv2.imshow("Depth Color Hand Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
