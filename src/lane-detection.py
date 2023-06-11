import matplotlib.pylab as plt
import cv2
import numpy as np


# Crops the frame to the region of interest, leaving only relevant data.
def crop_frame(frame):
    # Define width and height of frame.
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create polygon of 3 points based on frame dimensions.
    # This will create a triangle spanning from the bottom left and bottom right corners to the centrepoint of the image.
    relevant_polygon = [
        (0, frame_height),
        (frame_width / 2, frame_height / 2),
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


# Processes the frame, applying all necessary filters and transformations.
def process_frame(frame):
    # Apply crop
    frame = crop_frame(frame)

    return frame


# Select file or camera
cap = cv2.VideoCapture("media/video.mp4")

# Loop through frames
while cap.isOpened():
    ret, frame = cap.read()
    frame = process_frame(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
