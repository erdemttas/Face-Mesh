import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video14.mp4")

mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)

mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    print(results.multi_face_landmarks)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
    
        for id, lm in enumerate(faceLms.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            print([id,cx,cy])
            
            if id == 50:
                cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)
            
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, "FPS: "+str(int(fps)), (10,65), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        
    
    
    cv2.imshow("img", img)
    cv2.waitKey(10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    