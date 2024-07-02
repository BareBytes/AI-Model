import cv2
import matplotlib.pyplot as plt

config_file = r"C:\Users\srish\Downloads\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = r"C:\Users\srish\Downloads\frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'label.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read())

print(classLabels)
print(len(classLabels))

model.setInputSize(320,320)
model.setInputScale(1/127.5)
model.setInputMean(127.5)
model.setInputSwapRB(True)

cap = cv2.VideoCapture('Security_Guard_Footage trimmed_video.mp4')

# check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Can't open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, 0.50)
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if(ClassInd==1):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, font_scale, (0,255,0), 2)

    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #plt.imshow(frame_rgb)
    #plt.axis('off')
    #plt.show()

   # plt.pause(0.01)

    cv2.imshow('frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

#cap.release()
#cv2.destroyAllWindows()