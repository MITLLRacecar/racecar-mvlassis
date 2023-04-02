########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np
import enum # Για τη μηχανή καταστάσεων

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils
import ar_solver # Για την αναγνώριση των AR markers

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar() 

time = 0

# Κατάσταση που ορίζει τη φάση στην οποία βρισκόμαστε
class State(enum.IntEnum):
    start = 0
    line_following = 1
    lane_following = 2
    cone_slaloming = 3
    wall_following = 4
    done = 5

cur_state = State.start # Τρέχουσα φάση στην οποία βρισκόμαστε
cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32)) # Κενό marker

# Minimum number of pixels to consider a valid contour
MIN_CONTOUR_AREA = 50

# A crop window for the floor directly in front of the car
# CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
CROP_FLOOR = (
    (rc.camera.get_height() * 2 // 3, 0),
    (rc.camera.get_height(), rc.camera.get_width()),
)

# The HSV range for each color
# color[0] = low hsv limit
# color[1] = high hsv limit
# color[2] = color name
RED = ((170, 50, 50), (10, 255, 255), "RED")
GREEN = ((40, 50, 50), (80, 255, 255), "GREEN")
BLUE = ((100, 150, 150), (120, 255, 255), "BLUE")
ORANGE = ((10, 100, 100), (25, 255, 255), "ORANGE")
PURPLE = ((125, 100, 100), (150, 255, 255), "PURPLE")

# For phase 1
LINE_FOLLOW_SPEED = 1
# Προτεραιότητες χρωμάτων που θα ακολουθήσουμε στη φάση 1
colors = [PURPLE, ORANGE, RED, GREEN, BLUE]

# For phase 2
MIN_LANE_CONTOUR = 100
primary_color = ORANGE # Χρώμα που ορίζει τις ευθείες
secondary_color = PURPLE # Χρώμα που ορίζει τις απότομες στροφές
# Speed constants
LANE_FOLLOW_FAST_SPEED = 1
LANE_FOLLOW_SLOW_SPEED = 1
# Angle constants
# Amount to turn if we only see one lane
ONE_LANE_TURN_ANGLE = 1
MIN_LANE_CONTOUR_AREA = 200
HARD_TURN = False # True if we are currently making a hard_turn

def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_state
    global speed
    global angle

    # Have the car begin at a stop, in no_cones mode
    rc.drive.stop()
    cur_state = State.start
    speed = 0
    angle = 0

    # Print start message
    print(">> Time Trial Challenge")

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global cur_state
    global cur_marker
    global time 

    color_image = rc.camera.get_color_image()
    markers = rc_utils.get_ar_markers(color_image)

    time += rc.get_delta_time()

    # If we see a new marker, change the stage
    if len(markers) > 0 and markers[0].get_id() != cur_marker.get_id():
        cur_marker = markers[0]
        change_state()

    if cur_state == State.line_following:
        follow_lines()
    elif cur_state == State.lane_following:
        follow_lane()
    elif cur_state == State.cone_slaloming:
        cone_slalom()
    rc.drive.set_speed_angle(speed, angle)
    if 32 < time < 33.8:
        rc.drive.set_speed_angle(1, 0.2)
    if 33.8 < time < 34.7:
        rc.drive.set_speed_angle(1, 1)

# Καλείται κάθε 1 δευτερόλεπτο, χρησιμοποιείται για debugging (αποσφαλμάτωση)
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    print("Current state: ", cur_state)
    # print(colors)
    # print("Primary color is: ", primary_color[2])
    print(HARD_TURN)

# Μετάβαση στην επόμενη φάση και απαραίτητες ενέργειες για να προετοιμαστούμε
def change_state():
    global cur_state
    global colors
    global primary_color
    global secondary_color

    color_image = rc.camera.get_color_image_no_copy()
    markers = ar_solver.get_ar_markers(color_image)

    for marker in markers:
        print(str(marker))

    if cur_state == State.start:
        # Βρίσκουμε το marker που δείχνει αριστερά
        # και βάζουμε το χρώμα του πρώτο στην προτεραιότητα
        for marker in markers:
            if marker.get_orientation() == ar_solver.Orientation.LEFT:
                first_color = marker.get_color()
                for color in colors:
                    if first_color == color[2]: # color[2] = το όνομα του χρώματος
                        colors.remove(color)
                        colors.insert(0, color)
        cur_state = State.line_following
    elif cur_state == State.line_following:
        # Επιστρέφουμε την προτεραιότητα στο κανονικό
        first_color = colors[0]
        colors.remove(first_color)
        cur_state = State.lane_following
    elif cur_state == State.lane_following:
        for marker in markers:
            # Βρίσκουμε το χρώμα το marker
            # και ορίζουμε το πρωτεύον χρώμα με βάση αυτό
            if marker.get_color() == "PURPLE":
                primary_color = PURPLE
                secondary_color = ORANGE
                break
            elif marker.get_color() == "ORANGE":
                primary_color = ORANGE
                secondary_color = PURPLE
                break
            elif int(marker.get_id()) == 2 and marker.get_orientation() == ar_solver.Orientation.UP:
                corners = marker.get_corners()
                marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
                # Τραβάμε φωτογραφία βάθους
                depth_image = rc.camera.get_depth_image()
                # και βρίσκουμε την απόσταση του από εμάς
                marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
                print("marker_distance is: ", marker_distance)
                # Αν η απόσταση του marker από εμάς είναι μικρότερη από 100 εκατοστά
                if 0 < marker_distance < 100:
                    cur_state = State.cone_slaloming


# Για τη φάση 1: Line following όπως στο Lab 2A
def follow_lines():
    global cur_mode
    global speed
    global angle

    speed = LINE_FOLLOW_SPEED

    color_image = rc.camera.get_color_image()

    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])
   
    no_colors_found = True
    # Search for the colors in priority order
    for color in colors:
        # Find the largest contour of the specified color
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # If the contour exists, steer toward the center of the contour
        if contour is not None:
            no_colors_found = False
            center = rc_utils.get_contour_center(contour)
            # Το remap_range είναι πιο απότομο από ότι συνήθως
            # από το 1/4 της οθόνης και μετά στρίβουμε πλήρως
            angle = rc_utils.remap_range(center[1], rc.camera.get_width()*1//4, rc.camera.get_width()*3//4, -1, 1)
            angle = rc_utils.clamp(angle, -1, 1)
            rc_utils.draw_contour(cropped_image, contour)
            rc_utils.draw_circle(cropped_image, center)
            break
    if no_colors_found:
        # Αν δεν βρήκαμε χρώματα, προχωράμε ευθεία και ελπίζουμε
        angle = 0
    # Display the image to the screen
    rc.display.show_color_image(cropped_image)

# Για τη φάση 2: Lane following 
def follow_lane():
    global speed
    global angle
    global HARD_TURN
    global CROP_FLOOR

    speed = LANE_FOLLOW_SLOW_SPEED
    color_image = rc.camera.get_color_image()
    if color_image is None:
        # Έγινε λάθος και δεν βρήκαμε εικόνα
        print("No image")
        return
    markers = ar_solver.get_ar_markers(color_image)

    # Διαβάζουμε το marker που δείχνει τη σωστή στροφή στη δεύτερη φάση
    if len(markers) == 1 and markers[0].get_color() != primary_color[2]:
        # Παίρνουμε τις συντεταγμένες των 4 γωνιών του τετραγώνου
        corners = markers[0].get_corners()
        # Και με βάση αυτές βρίσκουμε το κέντρο του πάνω στην εικόνα
        marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
        # print(marker_center)
        # Τραβάμε φωτογραφία βάθους
        depth_image = rc.camera.get_depth_image()
        # και βρίσκουμε την απόσταση του από εμάς
        marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
        # print("marker_distance is: ", marker_distance)
        # Αν η απόσταση του marker από εμάς είναι μικρότερη από 100 εκατοστά
        if 0 < marker_distance < 100:
            # Στρίβουμε δεξιά αν το orientation marker του είναι προς τα δεξιά
            if corners[0][1] > corners[2][1]:
                angle = 0.5
            # Αλλιώς στρίβουμε αριστερά
            else:
                angle = -0.5
            return

    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])

    # Search for secondary color first
    contours = [
        contour
        for contour in rc_utils.find_contours(
            cropped_image, secondary_color[0], secondary_color[1]
        )
        if cv.contourArea(contour) > MIN_LANE_CONTOUR_AREA
    ]
    contours_primary = [
        contour
        for contour in rc_utils.find_contours(
            cropped_image, primary_color[0], primary_color[1]
        )
        if cv.contourArea(contour) > MIN_LANE_CONTOUR_AREA
    ]
    # Αν έχουμε μόνο δευτερεύον χρώμα βρισκόμαστε σε hard turn
    # στριψε μέχρι να βρούμε 2 γραμμές του πρωτεύοντος
    if len(contours) > 0 and len(contours_primary) == 0:
        print("seeing only secondary")
        HARD_TURN = True   
    if HARD_TURN == True:
        if len(contours_primary) >= 2:
            # Εδώ Βρήκαμε 2 γραμμές του πρωτεύοντος οπότε σταματάμε το hard turn
            print("done with turn")
            HARD_TURN = False
            CROP_FLOOR = (
                (rc.camera.get_height() * 3 // 4, 0),
                (rc.camera.get_height(), rc.camera.get_width()),
            )
        if len(contours) > 0:
            # Βρες
            contours.sort(key=cv.contourArea, reverse=True) # ταξινομούμε τα περιγράμματα
            contour = contours[0]
        if len(contours_primary) > 0:
            # Βρες το μεγαλύτερο περίγραμμα του πρωτεύοντος
            contours_primary.sort(key=cv.contourArea, reverse=True)
            contour_primary = contours_primary[0]
        if len(contours_primary) > 0 and (len(contours) == 0 or (len(contours) > 0 and rc_utils.get_contour_area(contour) < rc_utils.get_contour_area(contour_primary))):
            # Εδώ έχουμε μόνο 1 περίγραμμα του πρωτεύοντος
            center = rc_utils.get_contour_center(contour_primary)
            rc_utils.draw_contour(cropped_image, contour_primary)
            if center[1] > rc.camera.get_width() / 2:
                angle = 1
            else:
                angle = -1
        elif len(contours) > 0:
            # Εδώ έχουμε 1 περίγραμμα του δευτερεύοντος
            center = rc_utils.get_contour_center(contour)
            rc_utils.draw_contour(cropped_image, contour)
            if center[1] > rc.camera.get_width() / 2:
                # We can only see the RIGHT lane, so turn LEFT
                angle = -1
           
        
        # rc_utils.draw_circle(cropped_image, center)
        # Display the image to the screen
        rc.display.show_color_image(cropped_image)
        return
    if len(contours) == 0:
        # Secondary color not found, search for primary (fast) color
        # print("No secondary color found")
        contours = [
            contour
            for contour in rc_utils.find_contours(
                cropped_image, primary_color[0], primary_color[1]
            )
            if cv.contourArea(contour) > MIN_LANE_CONTOUR_AREA
        ]
        if len(contours) == 0:
            # print("No primary color found")
            speed = LANE_FOLLOW_SLOW_SPEED
            angle = 0
            follow_lines() # Δεν βρήκαμε κανένα από τα 2 χρώματα, άρα ψάξε για κάποιο από τα άλλα
            # rc.display.show_color_image(cropped_image)
            return
        else:
            speed = LANE_FOLLOW_FAST_SPEED

    if len(contours) >= 2:
        # Εδώ βρήκαμε τουλάχιστον δύο περιγράμματα
        # print("2 or more contours")
        # Τα ταξινομούμε με βάση το εμβαδόν
        contours.sort(key=cv.contourArea, reverse=True) # για να κρατήσουμε τα 2 μεγαλύτερα

        # Calculate the midpoint of the two largest contours
        first_center = rc_utils.get_contour_center(contours[0])
        second_center = rc_utils.get_contour_center(contours[1])
        # Βρίσκουμε το σημείο ανάμεσα στα 2 περιγράμματα
        midpoint = (first_center[1] + second_center[1]) / 2

        # Proportional control πάνω στο μέσο σημείο
        angle = rc_utils.remap_range(midpoint, 0, rc.camera.get_width(), -1, 1)

        # Draw the contours and centers onto the image (red one is larger)
        rc_utils.draw_contour(cropped_image, contours[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(cropped_image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(cropped_image, contours[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(cropped_image, second_center, rc_utils.ColorBGR.blue.value)
    else:
        # Εδώ βρήκαμε μόνο 1 περίγραμμα
        # Οπότε στρίβουμε προς την κατεύθυνση που περιμένουμε να είναι η άλλη γραμμή
        # που δεν βλέπουμε
        # print("1 contour")
        contour = contours[0]
        center = rc_utils.get_contour_center(contour)
        if center[1] > rc.camera.get_width() / 2:
            # We can only see the RIGHT lane, so turn LEFT
            angle = -ONE_LANE_TURN_ANGLE
        else:
            # We can only see the LEFT lane, so turn RIGHT
            angle = ONE_LANE_TURN_ANGLE
        # Draw the single contour and center onto the image
        rc_utils.draw_contour(cropped_image, contour)
        rc_utils.draw_circle(cropped_image, center)
    # Display the image to the screen
    rc.display.show_color_image(cropped_image)

# Για τη φάση 3: Cone slaloming
def cone_slalom():
    pass

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()