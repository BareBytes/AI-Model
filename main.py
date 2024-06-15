from ultralytics import YOLO
import cv2

# load the model
model = YOLO('yolov8n.pt')

# load the video
video_path = './Security_Guard_Footage.mp4'
cap = cv2.VideoCapture(video_path)

person_class_id = next(key for key, value in model.names.items() if value == 'person' )
ret = True

#unique_person_id = set()

# read frames
while ret:
    ret, frame = cap.read()

    # detect the object
    # track the object
    results = model.track(frame, persist=True)

    # only include person
    person_results = [res for res in results[0].boxes if res.cls == person_class_id and res.conf > 0.5]

    #for person in person_results:
    #    unique_person_id.add(person.id)

    results[0].boxes = person_results
    # Plot results
    frame_ = results[0].plot()

    # visualize
    cv2.imshow('frame',frame_)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break