'''
Created on 24 de jul de 2018

@author: nicoli
'''
import numpy as np
import cv2


from data_generator.batch_generator import set_test_image
from models.models import AlexNet
if __name__ == '__main__':
    
    n_classes=1
    img_width, img_height, img_depth = (224,224,3)
    alexnet = AlexNet(n_classes, img_width, img_height, img_depth, weights_path='/home/nicoli/github/deep-learning-learning/redes-tcc/callbacks/alexnet/age/class-weights-reg-fase2-2.30-104.07.hdf5')
    alexnet.model.compile(loss='mse', optimizer='adam', metrics=['mae'])
       
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.open(0)
    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier('/home/nicoli/opencv/opencv-3.4.1/data/haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image=gray, scaleFactor=2, minNeighbors=5)
        image_inputs = []
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            
            image_inputs.append(set_test_image(frame[y:y+h, x:x+w], 
                                         equalize=True,
                                         divide_by_stddev=255,
                                         resize=(img_height, img_width)))
        image_inputs = np.array(image_inputs).reshape(len(image_inputs), 224,224,3)
        if image_inputs.shape[0]!=0:
            print(alexnet.model.predict_on_batch(image_inputs))
        print(image_inputs.shape)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == 27: 
            break # esc to quit
    
cap.release()
cv2.destroyAllWindows()