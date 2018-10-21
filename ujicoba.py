import cv2
import timeit
import numpy
import traceback

from scipy import sqrt, pi, arctan2
from skimage.feature import hog
from skimage.exposure import rescale_intensity

def preProcessing(image):
    # Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Haar Feature & Crop
    face_cascade = cv2.CascadeClassifier('Haar Classifier XML/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    for (x, y, w, h) in faces:
        global roi_gray
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y + h, x:x + w]

    # Resize
    new_w, new_h = 64,128
    resize_image = cv2.resize(roi_gray, (new_w, new_h))
    print (resize_image.shape)
    return resize_image

def HOGPython(image) :
    hog_desc, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), block_norm='L2',visualise=True)
    return hog_desc,hog_image

def HOGBikin(image) :
    # Read image
    try :
        visualise = True
        image = numpy.atleast_2d(image)
        height,width = image.shape

        gx = numpy.zeros(image.shape)
        gy = numpy.zeros(image.shape)

        #robert
        #gx[:, :-1] = numpy.diff(image, n=1, axis=1)
        #gy[:-1, :] = numpy.diff(image, n=1, axis=0)

        #prewitt
        #for i in range(height):  # 128
        #    for j in range(width):  # 64
        #        if (i == 0) or (i == height - 1):
        #            gy[i, j] = 0
        #        else:
        #            gy[i, j] = image[(i + 1), j] - image[(i - 1), j]

        #for i in range(height):  # 128
        #    for j in range(width):  # 64
        #        if (j == 0) or (j == width - 1):
        #            gx[i, j] = 0
        #        else:
        #            gx[i, j] = image[i, (j + 1)] - image[i, (j - 1)]

        #magnitude = sqrt(gx ** 2 + gy ** 2)
        #angle = arctan2(gy, (gx + 1e-15)) * (180 / pi)


        #sobel
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)

        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        cell_size = 8
        block_size = 2
        orientations = 9
        sx, sy = image.shape
        cx, cy = (cell_size,cell_size)
        bx, by = (block_size,block_size)

        n_cellsx = int(numpy.floor(sx // cx))  # number of cells in x
        n_cellsy = int(numpy.floor(sy // cy))  # number of cells in y
        # compute orientations integral images
        orientation_histogram = numpy.zeros((n_cellsx, n_cellsy, orientations))
        temp_angle = numpy.zeros(shape=(sx,sy))
        temp_mag = numpy.zeros(shape=(sx,sy))
        for i in range(n_cellsx) : #baris
            for j in range (n_cellsy) : #kolom

                for o in range(orientations):
                    temp_bins = 0
                    for ii in range(i* cell_size,(i+1)*cell_size):
                        for jj in range(j * cell_size, (j + 1) * cell_size):
                            temp_angle[ii,jj] = numpy.where(angle[ii,jj] < 180 / orientations * (o + 1),
                                                angle[ii,jj], 0)
                            temp_angle[ii,jj] = numpy.where(angle[ii,jj] >= 180 / orientations * o,
                                                temp_angle[ii,jj], 0)
                            # select magnitudes for those orientations
                            cond2 = temp_angle[ii,jj] > 0
                            temp_mag[ii,jj] = numpy.where(cond2, magnitude[ii,jj], 0)
                            temp_bins += temp_mag[ii,jj]
                    #print("temp_angle cell baris",i," kolom ",j," bins ",o)
                    #print(temp_angle)
                    #print("temp_mag cell baris", i, " kolom ", j, " bins ", o)
                    #print(temp_mag)
                    orientation_histogram[i, j, o] = temp_bins
                    #print(orientation_histogram[i, j, o])

        hog_image = None

        if visualise:
            from skimage import draw

            radius = min(cx, cy) // 2 - 1
            orientations_arr = numpy.arange(orientations)
            # set dr_arr, dc_arr to correspond to midpoints of orientation bins
            orientation_bin_midpoints = (
                    numpy.pi * (orientations_arr + .5) / orientations)
            dr_arr = radius * numpy.sin(orientation_bin_midpoints)
            dc_arr = radius * numpy.cos(orientation_bin_midpoints)
            hog_image = numpy.zeros((sx, sy), dtype=float)
            for r in range(n_cellsx):
                for c in range(n_cellsy):
                    for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                        centre = tuple([r * cx + cx // 2,
                                        c * cy + cy // 2])
                        rr, cc = draw.line(int(centre[0] - dc),
                                           int(centre[1] + dr),
                                           int(centre[0] + dc),
                                           int(centre[1] - dr))
                        hog_image[rr, cc] += orientation_histogram[r, c, o]



        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        normalised_blocks = numpy.zeros((n_blocksx, n_blocksy,
                                      bx, by, orientations))

        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = orientation_histogram[x:(x + bx), y:(y + by), :]
                #print(block.shape)
                eps = 1e-5
                normalised_blocks[x, y, :] = block / numpy.sqrt(numpy.sum(block ** 2) + eps ** 2)


        if visualise:
            return normalised_blocks.ravel(), hog_image
        else:
            return normalised_blocks.ravel()
    except :
        print(traceback.format_exc())
        pass


image = cv2.imread('D:\\adhikagunadarma\\Kuliah\\TA\\TA\\Python\\skripsi-face-recognition\\PyCharm\\biasa.jpg')
print(image.shape)
resize = preProcessing(image)

#cv2.imshow("preprocess",resize)
tic = timeit.default_timer()
hog_feature,hog_image = HOGPython(resize)
hog_aing,hog_image_aing = HOGBikin(resize)
toc = timeit.default_timer()
print(toc-tic)
#print("Hog-aing = ",hog_aing.shape)
#for i in range(hog_aing.shape[0]):
#    print(hog_aing[i])

#print("Hog-feature = ",hog_feature.shape)
#for i in range(hog_feature.shape[0]):
#    print(hog_feature[i])

print("---------------------")
print("Hog-aing image= ",hog_image_aing.shape)
print(hog_image_aing)
print("Hog-feature image = ",hog_image.shape)
print(hog_image)

hog_image = cv2.resize(hog_image, (300, 600))
cv2.imshow("HOG Image before", hog_image)
hog_image_aing = cv2.resize(hog_image_aing, (300, 600))
cv2.imshow("HOG Image Aing before", hog_image_aing)

hogImage = rescale_intensity(hog_image, out_range=(0, 255))
hogImage = hogImage.astype("uint8")

hogImageAing = rescale_intensity(hog_image_aing, out_range=(0, 255))
hogImageAing = hogImageAing.astype("uint8")


hogImage = cv2.resize(hogImage, (300, 600))
cv2.imshow("HOG Image", hogImage)

hogImageAing = cv2.resize(hogImageAing, (300, 600))
cv2.imshow("HOG Image Aing", hogImageAing)


cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows