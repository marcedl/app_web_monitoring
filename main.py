import cv2
import time

video = cv2.VideoCapture(0) #0 means first/main camera
time.sleep(1)

first_frame = None

while True: 
    check, frame = video.read()
    #convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Gaussian bluur transformation to reduce noise (21, 21 is amount of bluur and 0 is std dev)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    
    #comparisson vs first frame
    if first_frame is None:
        first_frame = gray_frame_gau

    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
    
    #treshold for blacks and whites (above 30 is converted to 255)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    #to remove the noise we want to dilate it
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)
    #find contours and calculate the areas in all objects
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #small objects considered as fake objects
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("Video", frame)

    #key to stop the loop
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()