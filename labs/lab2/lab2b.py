"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 2B - Color Image Cone Parking
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

### Προσθήκη: βιβλιοθήκη για το enumeration του state machine
from enum import IntEnum
### Τέλος

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30

### Προσθήκη: Σταθερές που χρησιμοποιούμε
FORWARD_SPEED = 0.4
BACKWARD_SPEED = 0.4
BREAK_SPEED = 0.6
PARK_DOWN_THRESHOLD = 27100
PARK_UP_THRESHOLD = 27500
CLOSE_CONTOUR_AREA = 34000
ALIGN_ANGLE = 0.5
### Τέλος

# The HSV range for the color orange, stored as (hsv_min, hsv_max)
ORANGE = ((10, 100, 100), (20, 255, 255))

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

### Προσθήκη: προηγούμενο εμβαδόν του contour για να καταλαβαίνουμε την πραγματική
### Ταχύτητα του οχήματος
old_contour_area = 0 # The area of the previous contour
timer = 0
### Τέλος

### Προσθήκη: enumeration για το state machine
class State(IntEnum):
    straight = 1
    turn = 2
    obstacle = 3
    align = 4
    forward = 5
    backward = 6
    stop = 7

cur_state = State.straight
### Τέλος 

########################################################################################
# Functions
########################################################################################


def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the orange contours
        contours = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(image)


def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle

    ### Προσθήκη: Αρχικοποίηση των μεταβλητών που προσθέσαμε
    global cur_state
    cur_state = State.straight
    global timer
    timer = 0
    global old_contour_area
    old_contour_area = 0
    ### Τέλος

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(">> Lab 2B - Color Image Cone Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global cur_state
    global timer
    global old_contour_area

    old_contour_area = contour_area
    # Search for contours in the current color image
    update_contour()

    
    # TODO: Park the car 30 cm away from the closest orange cone
    ### Προσθήκη: Κώδικας για όλες τις καταστάσεις του state machine

    if cur_state == State.straight:
        speed = 1
        angle = 0
        cone_identified = contour_center != None
        #This
        about_to_hit_something = False
        if cone_identified and contour_area < PARK_DOWN_THRESHOLD:
            cur_state = State.forward
        elif cone_identified and contour_area >= PARK_DOWN_THRESHOLD:
            cur_state = State.backward
        elif about_to_hit_something:
            cur_state = State.obstacle
        else:
            timer += rc.get_delta_time()
            if timer > 3:
                cur_state = State.turn
                timer = 0

    elif cur_state == State.turn:
        speed = 0.2
        angle = -1
        timer += rc.get_delta_time()
        if timer > 3:
            cur_state = State.straight
            timer = 0

    elif cur_state == State.obstacle:
        speed = 1
        angle = 1
        # This
        obstacle_avoided = False # We assume there are no obstacles
        if obstacle_avoided:
            cur_state = State.straight
    # Πρέπει να μετακινηθούμε μπροστά
    elif cur_state == State.forward:
        speed = rc_utils.remap_range(contour_area, MIN_CONTOUR_AREA, PARK_DOWN_THRESHOLD, FORWARD_SPEED, 0.05)
        angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), -1, 1)
        if angle > ALIGN_ANGLE or angle < -ALIGN_ANGLE: # Ευθυγράμμιση αν δεν είμαστε σε σωστή γωνία
            cur_state = State.align
        elif contour_area >= PARK_DOWN_THRESHOLD:
            speed = -BREAK_SPEED
            angle = 0
            # Φρενάρισμα μόλις περάσουμε το κάτω όριο
            if contour_area <= old_contour_area:
                speed = 0
                if contour_area <= PARK_UP_THRESHOLD:
                    cur_state = State.stop
                else:
                    cur_state = State.backward
    # Πρέπει να μετακινηθούμε πίσω
    elif cur_state == State.backward:
        speed = rc_utils.remap_range(contour_area, CLOSE_CONTOUR_AREA, PARK_UP_THRESHOLD, -BACKWARD_SPEED, -0.05)
        angle = 0
        if contour_area <= PARK_UP_THRESHOLD:
            speed = BREAK_SPEED
            angle = 0
            # Φρενάρισμα μόλις περάσουμε το πάνω όριο
            if contour_area >= old_contour_area: 
                speed = 0
                if contour_area >= PARK_DOWN_THRESHOLD:
                    cur_state = State.stop
                else:
                    cur_state = State.forward

    ### Όταν δεν είμαστε ευθυγραμμισμένοι, όπισθεν μέχρι να ευθρυγραμμιστούμε
    elif cur_state == State.align:
        speed = -BACKWARD_SPEED
        angle = rc_utils.remap_range(contour_center[1], 0, rc.camera.get_width(), 1, -1)
        if -0.4 < angle < 0.4:
            cur_state = State.forward

    elif cur_state == State.stop:
        speed = 0
        angle = 0
        if contour_center == None:
            cur_state = State.straight
        if contour_area > PARK_UP_THRESHOLD:
            cur_state = State.backward
        elif contour_area < PARK_DOWN_THRESHOLD:
            cur_state = State.forward
    rc.drive.set_speed_angle(speed, angle)
    ### Τέλος

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)


def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    print(contour_area)
    print(cur_state)
    # Print a line of ascii text denoting the contour area and x position
    if rc.camera.get_color_image() is None:
        # If no image is found, print all X's and don't display an image
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # If an image is found but no contour is found, print all dashes
        if contour_center is None:
            print("-" * 32 + " : area = " + str(contour_area))

        # Otherwise, print a line of dashes with a | indicating the contour x-position
        else:
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print("".join(s) + " : area = " + str(contour_area))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
