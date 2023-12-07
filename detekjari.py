import cv2
import mediapipe as mp
import numpy as np

def count_fingers(frame, hand_contour):
    epsilon = 0.02 * cv2.arcLength(hand_contour, True)
    approx = cv2.approxPolyDP(hand_contour, epsilon, True)
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    finger_count = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])

        a = np.linalg.norm(np.array(far) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(end))
        c = np.linalg.norm(np.array(start) - np.array(end))
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))

        d = (2 * area) / c

        if d > 30 and far[1] < frame.shape[0] - 10:
            finger_count += 1
            cv2.circle(frame, far, 3, (0, 0, 255), -1)

    return finger_count

def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

def main():
    # Replace 'path/to/your/video.mp4' with the actual path to your video file
    video_path = 'video.mp4'
    cap = cv2.VideoCapture(video_path)

    # Initialize MediaPipe Hand module
    mp_hand = mp.solutions.hands
    hands = mp_hand.Hands()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error reading video file or end of video.")
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hand module
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                draw_landmarks(frame, hand_landmarks.landmark)

                # Convert hand landmarks to numpy array for contour calculation
                hand_np = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark])
                hand_np = hand_np.astype(np.int32)

                # Draw hand contour
                cv2.drawContours(frame, [hand_np], 0, (0, 255, 0), 2)

                # Count fingers
                finger_count = count_fingers(frame, hand_np)
                cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Finger Counter with Skeleton', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
