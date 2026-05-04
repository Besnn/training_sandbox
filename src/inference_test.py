import cv2
from ultralytics import YOLO


def main():
    model = YOLO('best.pt')

    video_path = ""
    image_path = ""

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Errore: Non riesco a trovare l'immagine in {image_path}")
        return

    results = model(frame, conf=0.2, imgsz=640)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv11 Image Test", annotated_frame)

    print("Premi un tasto qualsiasi sulla finestra dell'immagine per chiudere.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()