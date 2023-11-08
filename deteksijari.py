import cv2
import numpy as np

# Fungsi untuk mendeteksi ujung telunjuk dalam citra tangan
def detect_and_mark_finger_tips(frame):
    # Konversi frame ke citra grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Terapkan thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Temukan kontur dalam citra yang memiliki bentuk tangan
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Gambar garis tepi kontur pada citra asli
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Loop melalui setiap kontur (tangan) dan tandai ujung telunjuk dengan titik merah
    for contour in contours:
        if len(contour) >= 5:
            # Temukan titik tertinggi dalam kontur (ujung telunjuk)
            highest_point = tuple(contour[contour[:, :, 1].argmin()][0])

            # Gambar titik merah di ujung telunjuk
            cv2.circle(result, highest_point, 5, (0, 0, 255), -1)

    return result

# Buka video
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi ujung telunjuk dalam setiap frame video
    result_frame = detect_and_mark_finger_tips(frame)

    # Tampilkan hasilnya
    cv2.imshow("Finger Tip Detection", result_frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya
cap.release()
cv2.destroyAllWindows()
