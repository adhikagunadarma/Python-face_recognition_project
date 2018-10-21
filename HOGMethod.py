import cv2
import numpy
from scipy import sqrt, pi, arctan2, cos, sin
from skimage.feature import hog
from skimage.exposure import rescale_intensity


class HOGMethod :

    def __init__(self, cell_size, block_size, bins):
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins

    def hogPython(self,image,param):
        hog_desc, hog_image = hog(image, orientations=param.bins, pixels_per_cell=(param.cell_size, param.cell_size),
                                  cells_per_block=(param.block_size, param.block_size), block_norm='L2', visualise=True)
        return hog_desc, hog_image

    def hogCV(self,image,param):
        winSize = (param.resize_width, param.resize_height)
        blockSize = (param.block_size, param.block_size)
        blockStride = (param.cell_size, param.cell_size)
        cellSize = (param.cell_size, param.cell_size)
        nbins = param.bins
        signedGradient = param.signed_gradient


        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64

        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                                histogramNormType, L2HysThreshold, gammaCorrection, nlevels,signedGradient)
        h = hog.compute(image)
        return h

    def visualizeHog(self,image):
        hogImage = rescale_intensity(image, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")

        return hogImage


    def hogImage(self,image):
            visualise = True
            image = numpy.atleast_2d(image)

            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

            magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            cell_size = self.cell_size
            block_size = self.block_size
            orientations = self.bins
            sx, sy = image.shape
            cx, cy = (cell_size, cell_size)
            bx, by = (block_size, block_size)

            n_cellsx = int(numpy.floor(sx // cx))#8
            n_cellsy = int(numpy.floor(sy // cy))#16

            orientation_histogram = numpy.zeros((n_cellsx, n_cellsy, orientations))
            temp_angle = numpy.zeros(shape=(sx, sy))
            temp_mag = numpy.zeros(shape=(sx, sy))
            for i in range(n_cellsx):  # baris
                for j in range(n_cellsy):  # kolom
                    for o in range(orientations):
                        temp_bins = 0
                        for ii in range(i * cell_size, (i + 1) * cell_size):
                            for jj in range(j * cell_size, (j + 1) * cell_size):
                                temp_angle[ii, jj] = numpy.where(angle[ii, jj] < 180 / orientations * (o + 1),
                                                                 angle[ii, jj], 0)
                                temp_angle[ii, jj] = numpy.where(angle[ii, jj] >= 180 / orientations * o,
                                                                 temp_angle[ii, jj], 0)
                                cond2 = temp_angle[ii, jj] > 0
                                temp_mag[ii, jj] = numpy.where(cond2, magnitude[ii, jj], 0)
                                temp_bins += temp_mag[ii, jj]
                        # print("temp_angle cell baris",i," kolom ",j," bins ",o)
                        # print(temp_angle)
                        # print("temp_mag cell baris", i, " kolom ", j, " bins ", o)
                        # print(temp_mag)
                        orientation_histogram[i, j, o] = temp_bins
                        # print(orientation_histogram[i, j, o])

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


