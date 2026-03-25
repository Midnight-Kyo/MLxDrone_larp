"""Try every way to open the Insta360 Link 2 Pro camera."""
import cv2
import sys

CAMERA_NAME = "Insta360 Link 2 Pro"

attempts = [
    (f"video={CAMERA_NAME}", cv2.CAP_DSHOW, "DSHOW by name"),
    (f"video={CAMERA_NAME}:audio=none", cv2.CAP_FFMPEG, "FFMPEG by name"),
    (0, cv2.CAP_MSMF, "MSMF index 0"),
    (1, cv2.CAP_MSMF, "MSMF index 1"),
    (2, cv2.CAP_MSMF, "MSMF index 2"),
    (3, cv2.CAP_MSMF, "MSMF index 3"),
    (0, cv2.CAP_DSHOW, "DSHOW index 0"),
    (1, cv2.CAP_DSHOW, "DSHOW index 1"),
    (2, cv2.CAP_DSHOW, "DSHOW index 2"),
    (3, cv2.CAP_DSHOW, "DSHOW index 3"),
]

print(f"Trying to find: {CAMERA_NAME}\n")

for source, backend, label in attempts:
    try:
        cap = cv2.VideoCapture(source, backend)
    except Exception:
        cap = None
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            is_blank = frame.mean() < 5
            status = "BLANK/BLACK" if is_blank else "HAS IMAGE"
            print(f"  OK  [{label:20s}]  {w}x{h}  {status}")
            if not is_blank:
                cv2.putText(frame, f"{label} - {w}x{h}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(f"Found: {label}", frame)
                print(f"\n  >>> SHOWING PREVIEW. Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print(f"  FAIL [{label:20s}]  opened but can't read frame")
        cap.release()
    else:
        print(f"  FAIL [{label:20s}]  can't open")
        if cap:
            cap.release()

print("\nDone.")
