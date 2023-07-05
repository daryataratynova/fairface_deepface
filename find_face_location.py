from deepface import DeepFace 
from imutils import paths
import face_recognition
import time 
import cv2
import os
import csv
# pip install face_recognition

detectors = ['dlib']
threshold = 0.1

img_paths_125 = list(paths.list_files("fairface-img-margin125-trainval/val"))
img_paths_025 = list(paths.list_files("fairface-img-margin025-trainval/val"))
assert len(img_paths_025) == len(img_paths_125)

vis = False
image_counter = 0
face_counter = 0

t_start = time.time()
for model in detectors:
    found = False
    for img_path_025, img_path_125 in zip(img_paths_025, img_paths_125):
        # load the images
        img_025 = cv2.imread(img_path_025)
        img_125 = cv2.imread(img_path_125)
        
        # get facial encodings of the cropped face
        encoding_025 = face_recognition.face_encodings(img_025, [(0, img_025.shape[1], img_025.shape[0], 0)], num_jitters=1, model="large")[0]
        
        # detect faces on the large image
        results = DeepFace.extract_faces(img_125,
                                        target_size=(224, 224),
                                        detector_backend= model,
                                        enforce_detection=False,
                                        align=False,
                                        grayscale=False)
        
        for result in results:
            region = result["facial_area"]
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            confidence = result['confidence']
            if confidence > threshold:
                # extract encodings for the detected faces
                # compare it with the cropped face
                encoding_125 = face_recognition.face_encodings(img_125, [(y, w+x, h+y, x)], num_jitters=1, model="large")[0]
                similarity = face_recognition.compare_faces([encoding_125], encoding_025, tolerance=0.6)[0]
                print(similarity)
                    
                if vis:
                    cv2.rectangle(img_125, (x, y), (w+x, h+y), (0, 255, 0), 1)
                
                # if the detected face and cropped
                # are similar then increment the counter
                # and break the loop                 
                if similarity:
                    face_counter += 1
                    found = True
                    break
                found = False
        if not found:
            file  =  model+"_val_undetected.csv"
            file_exists = os.path.isfile(file)
            with open (file, 'a') as csvfile:
                headers = ['file']
                writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({'file': img_path_125})


                    
        if vis:
            cv2.imshow("img125", img_125)
            cv2.imshow("img025", img_025)
            if cv2.waitKey(0) & 0xFF == ord("q"):  
                break
            
        image_counter += 1
        print("[INFO] Processing image {}/{}. Accuracy: {:.2f}".format(image_counter, len(img_paths_125), face_counter/image_counter))
        
print("[INFO] Detected {}/{}. Accuracy: {:.2f}".format(face_counter, len(img_paths_125), face_counter/len(img_paths_125)))
t_end = time.time()
print("[INFO] Processing time: {:.2f} mins".format((t_start-t_end)/60))        
    
        
    
    
