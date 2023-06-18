import time
import matplotlib.pylab as plt
import cv2
import numpy as np


# Does what it says on the tin, converts a frame to grayscale.
def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, dstCn=2)


# Crops the frame to the region of interest, leaving only relevant data.
def crop_frame(frame):
    # Define width and height of frame.
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create polygon of 3 points based on frame dimensions.
    relevant_polygon = [
        (0, frame_height),
        (frame_width / 2, frame_height / 1.5),
        (frame_width, frame_height),
    ]

    # Create a mask of the same size as the frame, with all values set to 0.
    mask = np.zeros_like(frame)

    # Value to fill the polygon with. Black in this instance.
    matched_mask_color = 255

    # Fill the polygon with the mask color, creating a black triangle from the three points we had earlier
    cv2.fillPoly(mask, np.array([relevant_polygon], np.int32), matched_mask_color)

    # Mask the frame, leaving only the relevant triangle.
    masked_frame = cv2.bitwise_and(frame, mask)

    return masked_frame


# Detects edges in the frame using Canny edge detection.
def detect_edges(frame, threshold1, threshold2):
    frame = cv2.Canny(frame, threshold1, threshold2)

    return frame


# Use HoughLinesP to detect lines in the frame.
def detect_lines(frame):
    lines = cv2.HoughLinesP(
        frame,
        rho=1,
        theta=np.pi / 180,
        threshold=10,
        minLineLength=10,
        maxLineGap=100,
    )

    return lines


# Draw lines on the frame.
def draw_lines(original_frame, lines):
    # Create a blank image to draw lines on.
    canvas = np.zeros_like(original_frame)

    # If no lines were detected, return the original frame instead of crashing the program.
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)

        return cv2.addWeighted(original_frame, 0.5, canvas, 1, 0.0)
    except TypeError:
        return original_frame


# Apply blur to the frame, to reduce noise.
def apply_blur(frame, kernel_size):
    frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    return frame


# Apply thresholding to the frame, so only the white lines are shown.
# Should also include yellow lines in case of roadworks, but that's something for the future. ;)
def apply_thresholding(frame, lower, upper):
    ret, threshold = cv2.threshold(frame, lower, upper, cv2.THRESH_BINARY)
    frame = threshold
    return frame


# Processes the frame, applying all necessary filters and transformations.
def process_frame(frame):
    original_frame = np.copy(frame)

    # Convert to grayscale
    frame = convert_to_grayscale(frame)

    # Apply thresholding
    frame = apply_thresholding(frame, 120, 255)

    # Detect edges, before crop, because otherwise the cropped area outline will be seen as an edge
    frame = detect_edges(frame, 50, 255)

    # Apply crop
    frame = crop_frame(frame)

    # Detect lines
    lines = detect_lines(frame)

    # Draw lines on frame
    frame = draw_lines(original_frame, lines)

    return frame


# Select file or camera
cap = cv2.VideoCapture("media/video.mp4")

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    frame = process_frame(frame)
    cv2.imshow("frame", frame)

    # Show 20 frames per second, since that is the original framerate of the video.
    # time.sleep(1 / 20)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
