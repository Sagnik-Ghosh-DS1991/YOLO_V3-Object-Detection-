import cv2
import cv2.dnn
import numpy as np

cap = cv2.VideoCapture(0)

classnames=[]
whT=320
confidencethres=0.5
nms_thres=0.3

f=open(r'./coco.names.txt')
classnames=f.read().rstrip('\n').split('\n')
print(classnames)

model_configuration='./yolov3-320.cfg'
model_weights='./yolov3-320.weights'

net=cv2.dnn.readNetFromDarknet(model_configuration,model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_object(outputs,img):
    class_id=[]
    confidence=[]
    bbox=[]
    hT,wT,cT=img.shape
    for output in outputs:
        for det in output:
            score=det[5:]
            classids=np.argmax(score)
            confs=score[classids]
            if confs>confidencethres:
                w,h=int(det[2]*wT), int(det[3]*hT)
                x,y=int((det[0]* wT)-w/2),int((det[1] * hT)-h/2)
                bbox.append([x,y,w,h])
                confidence.append(float(confs))
                class_id.append(classids)
    print((bbox))
    indices=cv2.dnn.NMSBoxes(bbox,confidence,confidencethres,nms_thres)
    print(indices)
    for indx in indices:
        indx=indx[0]
        box=bbox[indx]
        x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
        cv2.putText(img,'{} {} % '.format(classnames[class_id[indx]].upper(),int(confidence[indx]*100)),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)
        #cv2.putText(img, f'{classnames[class_id[indx]].upper()} {int(confidence[indx] * 100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)



while True:
    outputlayers = []
    success, img = cap.read()
    blob_image=cv2.dnn.blobFromImage(img,1/255.0,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob_image)
    layer_names=net.getLayerNames()
    for i in net.getUnconnectedOutLayers():
        outputlayers.append(layer_names[i[0]-1])
    #print(outputlayers)
    outputs=net.forward(outputlayers)
    #print(len(outputs))
    #print(type(outputs))
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs)
    #print(net.getUnconnectedOutLayers())
    detect_object(outputs,img)




    cv2.imshow('Image', img)
    #print(img.shape)
    cv2.waitKey(1)

