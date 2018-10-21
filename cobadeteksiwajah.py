import cv2

path = "D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\PyCharm"

image = cv2.imread("s1.jpg")
# Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Haar Feature & Crop
face_cascade = cv2.CascadeClassifier('Haar Classifier XML/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image,1.3,5)

for (x, y, w, h) in faces:
        global roi_gray
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

    # Resize
new_w, new_h = 64,128
resize_image = cv2.resize(roi_gray, (new_w, new_h))

cv2.imshow('image',image)
cv2.imwrite('haar.png',image)
cv2.imshow('greyscale',gray_image)
cv2.imwrite('greyscale.png',gray_image)
#cv2.imshow('haar',roi_color)
cv2.imshow('haar',roi_gray)

cv2.imshow('resize',resize_image)
cv2.imwrite('resize.png',resize_image)


cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows