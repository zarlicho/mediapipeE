import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def menghitung_derajat(pertama,kedua,ketiga):
    pertama = np.array(pertama) 
    kedua = np.array(kedua) 
    ketiga = np.array(ketiga) 
    
    radians = np.arctan2(ketiga[1]-kedua[1], ketiga[0]-kedua[0]) - np.arctan2(pertama[1]-kedua[1], pertama[0]-kedua[0])
    derajat = np.abs(radians*180.0/np.pi)
    
    if derajat > 90: # dapat disesuaikan dengan maksimal pergerakan dari anggota badan tersebut
        derajat = 180-derajat
        
    return derajat


cap = cv2.VideoCapture('rick.mp4')
ret, frame = cap.read()
ih, iw, ic = frame.shape
crop = []
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
      
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
    
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        edge = np.zeros((512,512,3), np.uint8)
        
        try:
            landmarks = results.pose_landmarks.landmark
            mata_kanan = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y] # hanya memerlukan koordinat x dan y 
            hidung = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
            mata_kiri = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y]

            result_derajat = menghitung_derajat(mata_kanan, hidung, mata_kiri)
            a = tuple(np.multiply(mata_kiri,[770, 770]).astype(int)) #first 770, 770
            b = tuple(np.multiply(mata_kanan,[396, 281]).astype(int)) #end 396, 281
            c = tuple(np.multiply(hidung,[640, 0]).astype(int))
            x = (c[0])
            y = (c[1])

            print(iw, ih, ic)
            h, w = 310, 313
            # Ending coordinate, here (220, 220)
            # represents the bottom right corner of rectangle
            #end_point = ()
            # Blue color in BGR
            color = (255, 0, 0)
            crop = np.array(image[int(y):int(y + h),
                   int(x):int(x + w)])
            # Line thickness of 2 px
            thickness = 2
            xmin = (300)
            ymin = (264)
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            cv2.line(image, (300, 479), (300, 0), (255,0,0),2) #garis tengah
            cv2.line(image, (x, 638),(x, y),(0,0,255),2) #garis track
            cv2.line(image, (xmin, ymin),(x, 264),(0,0,255),2) #garis tengah
            cv2.circle(image, (x, y),(xmin, ymin),(0,0,255),2) #garis track
            cv2.putText(image, str(result_derajat), 
                           tuple(np.multiply(hidung,[750, 760]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
        except:
            pass
        mp_drawing.draw_landmarks(edge, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        #plt.imshow(image)
        #plt.show()
        cv2.imshow('Raw Webcam Feed', image)
        cv2.imshow('Raw', edge)
        cv2.imshow('crop', crop)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()