import cv2
import face_recognition

imgHR = face_recognition.load_image_file('Images/Hrithik Roshan.jpg')
imgHR = cv2.cvtColor(imgHR, cv2.COLOR_BGR2RGB)
faceLoc = face_recognition.face_locations(imgHR)[0]
encodeHR = face_recognition.face_encodings(imgHR)[0]
cv2.rectangle(imgHR, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,255), 2)

imgTest = face_recognition.load_image_file('Images/Hrithik Roshan Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeHRTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,255), 2)

results = face_recognition.compare_faces([encodeHR], encodeHRTest)
distance = face_recognition.face_distance([encodeHR], encodeHRTest)

print(results, distance)

cv2.putText(imgTest, f'{results} {round(distance[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

cv2.imshow('Hrithik Roshan',imgHR)
cv2.imshow('Hrithik Roshan Test',imgTest)
cv2.waitKey(0)
