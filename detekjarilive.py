import cv2
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
            cv2.circle(frame, far, 3, [0, 0, 255], -1)

    return finger_count

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_contour = max(contours, key=cv2.contourArea, default=None)

        if hand_contour is not None:
            cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)
            finger_count = count_fingers(frame, hand_contour)
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Finger Counter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
