import cv2

class Preprocessing :

    def __init__(self, resize_width, resize_height):
        self.resize_width = resize_width
        self.resize_height = resize_height

    # Grayscale
    def greyScale(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray_image

    # Haar Feature & Crop
    def haarFeatureClassifier(self,image) :

        face_cascade = cv2.CascadeClassifier('Haar Classifier XML/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, 1.3, 5)

        for (x, y, w, h) in faces:
            global roi_gray
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = image[y:y + h, x:x + w]
        return roi_gray

    #Resize Image
    def resizeImage(self,image) :
        resize_image = cv2.resize(image, (self.resize_width, self.resize_height))
        return resize_image