import cv2
from ultralytics import YOLO

def main():
    model_path = 'C:\\Users\\salih\\Desktop\\Face_Mask_Detection_using_YOLOv8m\\runs\\detect\\train5\\weights\\best.pt'
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time inference. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam. Exiting...")
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Real-Time YOLO Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
