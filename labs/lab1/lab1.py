"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 1 - Driving in Shapes
"""

########################################################################################
# Imports
########################################################################################

import sys

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Put any global variables here
### Προσθήκη: Λίστα από τις κινήσεις που πρόκειται να κάνουμε
moves_to_execute = []
### Τέλος

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Begin at a full stop
    rc.drive.stop()

    # Print start message
    # TODO (main challenge): add a line explaining what the Y button does
    print(
        ">> Lab 1 - Driving in Shapes\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = drive in a circle\n"
        "    B button = drive in a square\n"
        "    X button = drive in a figure eight\n"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # TODO (warmup): Implement acceleration and steering
    

    if rc.controller.was_pressed(rc.controller.Button.A):
        print("Driving in a circle...")
        # TODO (main challenge): Drive in a circle
        moves_to_execute.clear()
        circle_speed = 1
        circle_angle = 1
        circle_time = 5.5
        moves_to_execute.append([circle_speed, circle_angle, circle_time])

    # TODO (main challenge): Drive in a square when the B button is pressed
    if rc.controller.was_pressed(rc.controller.Button.B):
        print("Driving in a square...")
        # TODO (main challenge): Drive in a circle
        moves_to_execute.clear()
        line_speed = 1
        line_angle = 0
        line_time = 1.1
        corner_speed = 0.8
        corner_angle = 1
        corner_time = 1.65
        moves_to_execute.append([line_speed, line_angle, line_time])
        moves_to_execute.append([corner_speed, corner_angle, corner_time])
        moves_to_execute.append([line_speed/4, line_angle, line_time])
        moves_to_execute.append([corner_speed, corner_angle, corner_time+0.5])
        moves_to_execute.append([line_speed/4, line_angle, line_time])
        moves_to_execute.append([corner_speed, corner_angle, corner_time])
        moves_to_execute.append([line_speed/4, line_angle, line_time])
        moves_to_execute.append([corner_speed-0.1, corner_angle, corner_time])

    # TODO (main challenge): Drive in a figure eight when the X button is pressed
    if rc.controller.was_pressed(rc.controller.Button.X):
        print("Driving in a figure eight")
        moves_to_execute.clear()
        figure_eight_line_speed = 1
        figure_eight_line_angle = 0
        figure_eight_line_time = 3
        figure_eight_corner_speed = 0.5
        figure_eight_corner_angle = 1
        figure_eight_corner_time = 5.5
        moves_to_execute.append([figure_eight_line_speed, figure_eight_line_angle, figure_eight_line_time])
        moves_to_execute.append([figure_eight_corner_speed, figure_eight_corner_angle, figure_eight_corner_time])
        moves_to_execute.append([figure_eight_line_speed, figure_eight_line_angle, figure_eight_line_time])
        moves_to_execute.append([figure_eight_corner_speed, -figure_eight_corner_angle, figure_eight_corner_time])

    # TODO (main challenge): Drive in a shape of your choice when the Y button
    # is pressed
    if rc.controller.was_pressed(rc.controller.Button.Y):
        print("Driving in a heart shape")
        moves_to_execute.clear()
        heart_line_speed = 1
        heart_line_angle = 0.4
        heart_line_time = 2
        heart_curve_speed = 0.4
        heart_curve_angle = 1
        heart_curve_time = 5
        moves_to_execute.append([heart_line_speed, heart_line_angle, heart_line_time])
        moves_to_execute.append([heart_curve_speed, heart_curve_angle, heart_curve_time])
        moves_to_execute.append([0.4, -1, 2.5])
        moves_to_execute.append([heart_curve_speed, heart_curve_angle, heart_curve_time+1.7])
        moves_to_execute.append([heart_line_speed, heart_line_angle, heart_line_time-0.8])
    # Αν υπάρχουν κινήσεις που δεν έχουμε εκτελέσει
    if moves_to_execute:
        # Πάρε την κίνηση που βρίσκεται στο τέλος και θέσε την κατάλληλη ταχύτητα/γωνία
        rc.drive.set_speed_angle(moves_to_execute[0][0], moves_to_execute[0][1])    
        # Αφαίρεσε τον χρόνο που πέρασε
        moves_to_execute[0][2] -= rc.get_delta_time()
        # Αν τελείωσε ο χρόνος που θέλαμε για τη συγκεκριμένη, τότε διάγραψε την κίνηση και προχώρα στην επόμενη
        if moves_to_execute[0][2] <= 0:
            moves_to_execute.pop(0)
    # Διαφορετικά, απλά όρισε την ταχύτητα από το controller
    else:
        forward_speed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
        backward_speed = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        (angle, _) = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
        rc.drive.set_speed_angle(forward_speed - backward_speed, angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
