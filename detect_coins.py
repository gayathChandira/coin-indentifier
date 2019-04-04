import math
import numpy as np
import argparse
import cv2


# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# resize image while retaining aspect ratio
d = 1024 / image.shape[1]
dim = (1024, int(image.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# create a copy of the image to display results
output = image.copy()

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# improve contrast accounting for differences in lighting conditions:
# clahe = Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)

# blur the image using Gaussian blurring, where pixels closer to the center
# contribute more "weight" to the average, first argument is the source image,
# second argument is kernel size, third one is sigma (0 for autodetect)
# we use a 7x7 kernel and let OpenCV detect sigma
#blurred = cv2.GaussianBlur(cl1, (5, 5), 0)
blurred = cv2.medianBlur(cl1, 5);
#cv2.HoughCircles will detect circles in the picture.
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                           param1=200, param2=100, minRadius=50, maxRadius=120)


diameter = []
materials = []
coordinates = []

count = 0
if circles is not None:
    # append radius to list of diameters (we don't bother to multiply by 2)
    for (x, y, r) in circles[0, :]:
        diameter.append(r)

    # convert coordinates and radii to integers
    circles = np.round(circles[0, :]).astype("int")
    #print(circles)
    # loop over coordinates and radii of the circles
    for (x, y, d) in circles:
        count += 1
        # add coordinates to list
        coordinates.append((x, y))
        # extract region of interest
        roi = image[y - d:y + d, x - d:x + d]
        # draw contour and results in the output image
        cv2.circle(output, (x, y), d, (0, 255, 0), 2)


# get biggest diameter
biggest = max(diameter)
i = diameter.index(biggest)
print(materials)
print('Maximum diameter: ', biggest)

# scale everything according to maximum diameter
diameter = [x / biggest * 25 for x in diameter]


i = 0

while i < len(diameter):
    d = diameter[i]
    (x, y) = coordinates[i]
    t = "Unknown"

    if math.isclose(d, 25.00, abs_tol=1.00):
        t = "Rs.2"

    elif math.isclose(d, 22.55, abs_tol=1.45):
        t = "Rs.10"

    elif math.isclose(d, 19.85, abs_tol=1.25):
        t = "Rs.5"

    elif math.isclose(d, 17.00, abs_tol=1.5):
        t = "Rs.1"


    # write result on output image
    cv2.putText(output, t,
                (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN,
                1.5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(output, str(d),
                (x - 40, y + 40), cv2.FONT_HERSHEY_PLAIN,
                1.5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    i += 1

# resize output image while retaining aspect ratio
d = 600 / output.shape[1]
dim = (768, int(output.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)


# show output and wait for key to terminate program
cv2.imshow("Output", np.hstack([image, output]))
cv2.waitKey(0)
