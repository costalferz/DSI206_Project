import os 
import numpy as np
import cv2
import imutils


# Constants
FILES_DIR = "./" # Where we will put the temp files
    
def read_image(input_image):
    """
    Function to read an image and return OpenCV object
    """
    try:
        # Read the image with OpenCV
        image = cv2.imread(input_image)
    except AttributeError: 
        print(f"Your input file '{input_image}' doesn't seems to be a valid.")
    except:
        print("Unknown error, sorry.")
        
    return image

    
def show_image_opencv(image_instance, name="Image in OpenCV"):
    """
    Function to show an image in OpenCV popup.
    It is possible to have some problems in *nix systems.
    """
    try:
        cv2.imshow(name, image_instance)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Unknown error, sorry.")

def save_image_opencv(image_instance, target_name=os.path.join(FILES_DIR, "result.jpg")):
    """
    Save a file from OpenCV image instance.
    """
    try:
        cv2.imwrite(target_name, image_instance)
    except:
        print(f"Unknown error, sorry. Your provided instance: {image_instance} with target: {target_name}")
    
# Get the input image in OpenCV object
input_image = read_image(os.path.join(FILES_DIR, "receipt-scanned.jpg"))
# Make a copy of the image
original_image = input_image.copy()

# Save the image, even there is no big sense doing so in current stage (only to show it like expected result)
save_image_opencv(input_image, os.path.join(FILES_DIR, "input_image.jpg"))

def detect_edges(input_image):
    """
    Function to return an edged image from input of normal OpenCV image
    """

    # Convert the image to gray scale
    # On that way we should be able to remove color noise
    # https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Blur the image to remove the high frequency noise 
    # This will help us with the task for find and detect contours in the gray image (we made above)
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#averaging
    gray_image_blured = cv2.blur(gray_image, (3, 3))

    # Perform Canny edge detection
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
    edged_image = cv2.Canny(gray_image_blured, 100, 400, 3)

    return edged_image

# Use our function and perform edge detection to input image
edged_image = detect_edges(input_image)

# Saving the image in order to show it below for demo purposes
save_image_opencv(edged_image, os.path.join(FILES_DIR, "edged_image.jpg"))

def calculate_draw_contours(edged_image, target_image): 
    """
    Function to calculate and draw the contours.
    """
    # Find the contours
    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
    all_contours = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = imutils.grab_contours(all_contours)

    # Make sort by contourArea and get the largest element. Sort in reverse.
    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#contourarea
    all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[:1]

    # Calculates a contour perimeter or a curve length.
    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#arclength
    contour_perimeter = cv2.arcLength(all_contours[0], True) 
    
    # Approximates a polygonal curve(s) with the specified precision.
    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#approxpolydp
    approximated_poly = cv2.approxPolyDP(all_contours[0], 0.02 * contour_perimeter, True)

    # Draw the contours to the target image
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#how-to-draw-the-contours
    cv2.drawContours(target_image, [approximated_poly], -1, (0,255,0), 2)
    
    return approximated_poly, contour_perimeter


# Use the function to draw our contours
approximated_poly, contour_perimeter = calculate_draw_contours(edged_image, input_image)

# Saving the image in order to show it below for demo purposes
save_image_opencv(input_image, os.path.join(FILES_DIR, "contoured_image.jpg"))

# Reshape the coordinates array
approximated_poly = approximated_poly.reshape(4, 2)

# A list to hold coordinates
rectangle = np.zeros((4, 2), dtype="float32")
                
# Top left corner should contains the smallest sum, 
# Bottom right corner should contains the largest sum
s = np.sum(approximated_poly, axis=1)
rectangle[0] = approximated_poly[np.argmin(s)]
rectangle[2] = approximated_poly[np.argmax(s)]

# top-right will have smallest difference
# botton left will have largest difference
diff = np.diff(approximated_poly, axis=1)
rectangle[1] = approximated_poly[np.argmin(diff)]
rectangle[3] = approximated_poly[np.argmax(diff)]

# Top left (tl), Top right (tr), Bottom right (br), Bottom left (bl)
(tl, tr, br, bl) = rectangle

def calculate_max_width_height(tl, tr, br, bl):
    """
    Function to calculate max width and height.
    Accepting the coordinates.
    """
    # Calculate width
    width_a = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    width_b = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    max_width = max(int(width_a), int(width_b))

    # Calculate height
    height_a = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    height_b = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    max_height = max(int(height_a), int(height_b))
    
    return max_width, max_height

max_width, max_height = calculate_max_width_height(tl, tr, br, bl)

 # Set of destinations points
# Dimensions of the new image
destinations = np.array([
        [0,0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

# Calculates a perspective transform from four pairs of the corresponding points.
# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#getperspectivetransform
transformation_matrix = cv2.getPerspectiveTransform(rectangle, destinations)

def apply_transformation(image_instance, transformation_matrix, max_width, max_height):
    # Applies a perspective transformation to an image
    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective
    scan = cv2.warpPerspective(image_instance, transformation_matrix, (max_width, max_height))
    return scan

# Apply the transformation from our function
scanned_image = apply_transformation(original_image, transformation_matrix, max_width, max_height)

# Save the temp files
save_image_opencv(scanned_image, os.path.join(FILES_DIR, "scanned_image.jpg"))


cv2.imshow('original_image',original_image)
cv2.imshow('input_image',input_image)
cv2.imshow('edge_image',edged_image)
cv2.imshow('scanned',scanned_image)
cv2.waitKey(0)