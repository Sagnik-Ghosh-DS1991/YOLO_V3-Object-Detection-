from flask import Flask, jsonify,request,Response,Request,render_template,url_for,redirect
import readchar
import cv2
import numpy as np

app=Flask(__name__)

classnames = []
whT = 320
confidencethres = 0.5
nms_thres = 0.3

f = open(r'./models/coco.names.txt')
classnames = f.read().rstrip('\n').split('\n')
model_configuration = './models/yolov3-320.cfg'
model_weights = './models/yolov3-320.weights'

net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
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
    indices=cv2.dnn.NMSBoxes(bbox,confidence,confidencethres,nms_thres)
    for indx in indices:
        indx=indx[0]
        box=bbox[indx]
        x,y,w,h=box[0],box[1],box[2],box[3]
        text = classnames[class_id[indx]].upper() + ': ' + str(int(confidence[indx]*100)) + ' %'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 3)
        text_w, text_h = text_size[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),(45,255,255),3)
        cv2.rectangle(img, (x, y), (x + text_w, y - 50), (45,255,255),-1)
        cv2.putText(img,'{}: {} % '.format(classnames[class_id[indx]].upper(),int(confidence[indx]*100)),(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)

def gen_frames():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error Opening Webcam")
    while (cap.isOpened()):
        #cap.set(3,640)
        #cap.set(4,480)
        outputlayers = []
        success, img = cap.read()
        if not success:
            break
        else:
            blob_image = cv2.dnn.blobFromImage(img, 1 / 255.0, (whT, whT), [0, 0, 0], 1, crop=False)
            net.setInput(blob_image)
            layer_names = net.getLayerNames()
            for i in net.getUnconnectedOutLayers():
                outputlayers.append(layer_names[i[0] - 1])
            outputs = net.forward(outputlayers)
            detect_object(outputs, img)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/LiveStreamingDetect',methods=['GET','POST'])
def LiveStreamingDetect():
    if (request.method == 'GET'):
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        cv2.VideoCapture(0).release()
        return render_template('index.html')


@app.route('/YoloObjectDetect',methods=['GET','POST'])
def Yolo_object_detect():
    if (request.method=='POST'):
        return render_template('detect_yolov3_object.html')
    else:
        cv2.VideoCapture(0).release()
        return render_template('index.html')


@app.route('/',methods=['GET'])
def Yolo_homepage():
    return render_template('index.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)