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

# Κατάσταση που ορίζει τη φάση στην οποία βρισκόμαστε
class State(enum.IntEnum):
    phase_1 = 0
    phase_2 = 1
    phase_3 = 2
    phase_4 = 3
    phase_5 = 4
    phase_6 = 5
    phase_7 = 6
    phase_8 = 7
    phase_9 = 8
    recover = 9 # Για αν κολλήσει το racecar

# Κατάσταση στη φάση 2 (wall following)    
class Wall_Mode(enum.IntEnum):
    align = 0
    right_safety = 1
    left_safety = 2
    reverse = 3
    
# Κατάσταση στη φάση 6 (cone line following)
class Cone_Mode(enum.IntEnum):
    align = 0
    right_safety = 1
    left_safety = 2

cur_state = State.phase_1 # Τρέχουσα φάση στην οποία βρισκόμαστε
cur_wall_mode = Wall_Mode.align
cur_cone_mode = Cone_Mode.align # Αρχική κατάσταση του cone_mode
cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32)) # Κενό marker

# Ο ελάχιστος αριθμός pixels που απαιτούνται για να αναγνωρισθεί ένα
# περίγραμμα
MIN_CONTOUR_AREA = 30

# The HSV range for each color
# color[0] = low hsv limit
# color[1] = high hsv limit
# color[2] = color name
GREEN = ((40, 50, 50), (80, 255, 255), "GREEN")
ORANGE = ((10, 100, 100), (25, 255, 255), "ORANGE")
PURPLE = ((125, 100, 100), (150, 255, 255), "PURPLE")

### Για τη συνάρτηση follow_lines()
# Προτεραιότητες χρωμάτων που θα ακολουθήσουμε, εδώ μας ενδιαφέρει μόνο η
# πράσινη γραμμή
colors = [GREEN]
# Το παράθυρο που θα κρατάμε κάθε φορά μετά το crop, ορίζεται από το πάνω
# αριστερά και το κάτω δεξιά σημείο. Κάθε σημείο είναι της μορφής (row, column)
CROP_FLOOR = (
    (rc.camera.get_height() * 55 // 100, 0),
    (rc.camera.get_height(), rc.camera.get_width())
)
LINE_FOLLOW_SPEED = 1 # Ταχύτητα με την οποία ακολουθούμε τις γραμμές
MAX_LINE_DISTANCE = 100 # Μέγιστη απόσταση από την οποία θα ακολουθούμε μια γραμμή


### Για το follow_walls()
SAFETY_DISTANCE = 30
# When a wall is greater than this distance (in cm) away, exit safety mode
END_SAFETY_DISTANCE = 32
# Speed to travel in safety mode
SAFETY_SPEED = 0.2
# The minimum and maximum angles to consider when measuring closest side distance
MIN_SIDE_ANGLE = 10
MAX_SIDE_ANGLE = 60
# The angles of the two distance measurements used to estimate the angle of the left
# and right walls
SIDE_FRONT_ANGLE = 70
SIDE_BACK_ANGLE = 110
WINDOW_ANGLE = 12
TURN_THRESHOLD = 20
# Για λίγο χρονικό διάστημα στην αρχή θέλουμε να πηγαίνουμε απλά ευθεία μέχρι να φτάσουμε στους τοίχους
FOLLOW_WALLS_START_LENGTH = 0.4
follow_walls_counter = 0

### Για το follow_lane()
CROP_FLOOR_2 = (
    (rc.camera.get_height() * 3 // 5, 0),
    (rc.camera.get_height(), rc.camera.get_width())
) 
primary_color = ORANGE # Χρώμα που ορίζει τις ευθείες
secondary_color = PURPLE # Χρώμα που ορίζει τις απότομες στροφές
# Speed constants
LANE_FOLLOW_FAST_SPEED = 0.8
LANE_FOLLOW_SLOW_SPEED = 0.5
# Angle constants
# Amount to turn if we only see one lane
ONE_LANE_TURN_ANGLE = 1
MIN_LANE_CONTOUR_AREA = 80


### Για τη συνάρτηση follow_lines_cone()
CROP_FLOOR_3 = (
    (rc.camera.get_height() *  80 // 100, rc.camera.get_width() * 1 // 6),
    (rc.camera.get_height(), rc.camera.get_width() * 5 // 6)
)
CONE_FOLLOW_SPEED = 0.9


# Για το lidar (χρησιμοποιείται στη συνάρτηση follow_lines_cone())
# >> Constants
# The maximum speed the car will travel
# When an object in front of the car is closer than this (in cm), start braking
BRAKE_DISTANCE = 40
BRAKE_MAX_ANGLE = 0.7
# When a wall is within this distance (in cm), focus solely on not hitting that wall
SAFETY_DISTANCE = 21
# When a wall is greater than this distance (in cm) away, exit safety mode
END_SAFETY_DISTANCE = 22
# Speed to travel in safety mode
SAFETY_SPEED = 0.2
# The minimum and maximum angles to consider when measuring closest side distance
MIN_SIDE_ANGLE = 10
MAX_SIDE_ANGLE = 70

### Για την check_surroundings() και τη recover()
save_state = State.phase_1
recovery_counter = 0
recovery_time = 1 # Χρόνος που κάνουμε recover
RECOVERY_SPEED = -1 # Ταχύτητα με την οποία κάνουμε recover
RECOVERY_ANGLE = 0 # Γωνία με την οποία κάνουμε recover

def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_state
    global speed
    global angle

    # Ξεκινάμε ακίνητοι
    rc.drive.stop()
    
    # Η παρακάτω συνάρτηση αυξάνει τη ΜΕΓΙΣΤΗ ταχύτητα του αυτοκινήτου
    # Πηγαίνει από 0 μέχρι 1, by default είναι 0.25
    # Την αλλάζουμε με πολλή προσοχή
    rc.drive.set_max_speed(0.25)
    
    # Ξεκινάμε στο mode που επιθυμούμε
    cur_state = State.phase_1
    speed = 0
    angle = 0

    # Print start message
    print(">> Final Challenge - Jumpscar - Charalambos Kokkinos, Dimitris Panagiotakopoulos, Vasilis Petropoulos, Konstantinos Stavrou")

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global cur_state
    global cur_marker

    color_image = rc.camera.get_color_image()
    markers = rc_utils.get_ar_markers(color_image)

    # If we see a new marker, change the stage
    # if len(markers) > 0 and markers[0].get_id() != cur_marker.get_id():
    #     cur_marker = markers[0]
    #     change_state()
    if cur_state == State.phase_1:
        follow_lines(0.85, 0.7)
    elif cur_state == State.phase_2:
        follow_walls(0.4, 0.9)
    elif cur_state == State.phase_3:
        follow_lane(0.44, 0.9)
    elif cur_state == State.phase_4:
        follow_lines(0.6)
    elif cur_state == State.phase_5:
        follow_lines(0.45)
    elif cur_state == State.phase_6:
        follow_lines_cone(0.22, 1, CROP_FLOOR_3)
    elif cur_state == State.phase_7:
        follow_lines(0.57)
        # pass_trains(0.5)
    elif cur_state == State.phase_8:
        follow_lines(0.4, 1)
        ang_vel = rc.physics.get_angular_velocity()
        if ang_vel[2] < -0.04:
            speed = -1
    elif cur_state == State.phase_9:
        follow_lines(0.45, 0.75)
    elif cur_state == State.recover:
        recover()

    change_state()        
    rc.drive.set_speed_angle(speed, angle)
    check_surroundings()

# Καλείται κάθε 1 δευτερόλεπτο, χρησιμοποιείται για debugging (αποσφαλμάτωση)
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # print("Current state:", cur_state)
    # if (cur_state == State.phase_2):
    #     print("Current mode:", cur_wall_mode)

# Μετάβαση στην επόμενη φάση και απαραίτητες ενέργειες για να προετοιμαστούμε
def change_state():
    global cur_state
    global colors
    global primary_color
    global secondary_color

    color_image = rc.camera.get_color_image_no_copy()
    markers = ar_solver.get_ar_markers(color_image)

    for marker in markers:
        # print(str(marker))
        pass
        
    if cur_state == State.phase_1:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 0:
                corners = marker.get_corners()
                marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
                depth_image = rc.camera.get_depth_image()
                marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
                if marker_distance < 100:
                    cur_state = State.phase_2
    elif cur_state == State.phase_2:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 1:
                cur_state = State.phase_3
                # Ορίζουμε το primary color με βάση το χρώμα του marker
                if marker.get_color() == "PURPLE":
                    primary_color = PURPLE
                    secondary_color = ORANGE
                elif marker.get_color() == "ORANGE":
                    primary_color = ORANGE
                    secondary_color = PURPLE      
    elif cur_state == State.phase_3:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 0 and marker.get_orientation() == ar_solver.Orientation.UP:
                
                cur_state = State.phase_4
    elif cur_state == State.phase_4:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 3 and marker.get_orientation() == ar_solver.Orientation.UP:
                cur_state = State.phase_5
    elif cur_state == State.phase_5:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 4 and marker.get_orientation() == ar_solver.Orientation.UP:
                cur_state = State.phase_6
    elif cur_state == State.phase_6:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 5 and marker.get_orientation() == ar_solver.Orientation.UP:
                cur_state = State.phase_7
    elif cur_state == State.phase_7:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 6 and marker.get_orientation() == ar_solver.Orientation.UP:
                cur_state = State.phase_8
    elif cur_state == State.phase_8:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 8 and marker.get_orientation() == ar_solver.Orientation.UP:
                cur_state = State.phase_9


# Line following όπως στο Lab 2A
def follow_lines(max_speed=0.25, sensitivity=1, crop_floor=CROP_FLOOR):
    rc.drive.set_max_speed(max_speed)
    global cur_mode
    global speed
    global angle

    speed = LINE_FOLLOW_SPEED
    
    color_image = rc.camera.get_color_image()
    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, crop_floor[0], crop_floor[1])

    # Depth image για να αγνοούμε κάποια γραμμή αν είναι πολύ μακριά
    depth_image = rc.camera.get_depth_image()
    cropped_depth_image = rc_utils.crop(depth_image, crop_floor[0], crop_floor[1])

    # Ψάχνουμε για περιγράμματα στην προτεραιότητα που ορίζεται από το colors
    for color in colors:
        # Βρίσκουμε όλα τα περιγράμματα του συγκεκριμένου χρώματος που υπάρχουν
        # στην κροπαρισμένη εικόνα
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        # Και από αυτά παίρνουμε το μεγαλύτερο (αρκεί να είναι μεγαλύτερο από το
        # MIN_CONTOUR_AREA        
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # If the contour exists, steer toward the center of the contour
        if contour is not None:
            center = rc_utils.get_contour_center(contour)
            contour_distance = rc_utils.get_pixel_average_distance(cropped_depth_image,
                                                                   center)
            # Αν η απόσταση της γραμμής είναι πολύ μεγάλη, τότε αγνόησε την
            if (contour_distance > MAX_LINE_DISTANCE):
                continue
            
            # Το remap_range είναι πιο απότομο από ότι συνήθως
            angle = rc_utils.remap_range(center[1], rc.camera.get_width()*1//3, rc.camera.get_width()*2//3, -sensitivity, sensitivity, True)
            rc_utils.draw_contour(cropped_image, contour, rc_utils.ColorBGR.yellow.value)
            rc_utils.draw_circle(cropped_image, center)
            break
    # Προβάλουμε την κροπαρισμένη εικόνα
    rc.display.show_color_image(cropped_image)

# Line following, απλά πρέπει να πήγαίνουμε αργά για να αποφύγουμε τους κώνους
# και να στρίβουμε από την άλλη αν πάμε να χτυπήσουμε κάποιον
def follow_lines_cone(max_speed = 0.25, sensitivity=1, crop_floor = CROP_FLOOR):
    rc.drive.set_max_speed(max_speed)
    global cur_mode
    global speed
    global angle
    global cur_cone_mode

    speed = CONE_FOLLOW_SPEED
    
    color_image = rc.camera.get_color_image()
    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, crop_floor[0], crop_floor[1])

    # Depth image για να αγνοούμε κάποια γραμμή αν είναι πολύ μακριά
    depth_image = rc.camera.get_depth_image()
    cropped_depth_image = rc_utils.crop(depth_image, crop_floor[0], crop_floor[1])

    # Ψάχνουμε για περιγράμματα στην προτεραιότητα που ορίζεται από το colors
    for color in colors:
        # Βρίσκουμε όλα τα περιγράμματα του συγκεκριμένου χρώματος που υπάρχουν
        # στην κροπαρισμένη εικόνα
        contours = rc_utils.find_contours(cropped_image, color[0], color[1])
        # Και από αυτά παίρνουμε το μεγαλύτερο (αρκεί να είναι μεγαλύτερο από το
        # MIN_CONTOUR_AREA        
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        # If the contour exists, steer toward the center of the contour
        if contour is not None:
            center = rc_utils.get_contour_center(contour)
            contour_distance = rc_utils.get_pixel_average_distance(cropped_depth_image,
                                                                   center)
            # Αν η απόσταση της γραμμής είναι πολύ μεγάλη, τότε αγνόησε την
            if (contour_distance > MAX_LINE_DISTANCE):
                continue
            
            # Το remap_range είναι πιο απότομο από ότι συνήθως
            angle = rc_utils.remap_range(center[1], 0, crop_floor[1][1] - crop_floor[0][1], -sensitivity, sensitivity, True)
            angle = rc_utils.clamp(angle, -1, 1)
            rc_utils.draw_contour(cropped_image, contour, rc_utils.ColorBGR.yellow.value)
            rc_utils.draw_circle(cropped_image, center)
            break

    # Βρήκαμε την κατάλληλη γωνία, αλλά αν είμαστε πολυ κοντά σε κώνο μπαίνουμε σε left_safety ή right_safety
    scan = rc.lidar.get_samples()
    left_angle, left_dist = rc_utils.get_lidar_closest_point(
        scan, (-MAX_SIDE_ANGLE, -MIN_SIDE_ANGLE)
    )
    right_angle, right_dist = rc_utils.get_lidar_closest_point(
        scan, (MIN_SIDE_ANGLE, MAX_SIDE_ANGLE)
    )
    # Αν είμαστε αρκετά κοντά σε εμπόδιο, μην στρίβεις τόσο απότομα
    # κάνε clamp (Lab 2) την τιμή
    if left_dist < BRAKE_DISTANCE:
        angle = rc_utils.clamp(angle, -BRAKE_MAX_ANGLE, BRAKE_MAX_ANGLE)
    if right_dist < BRAKE_DISTANCE:
        angle = rc_utils.clamp(angle, -BRAKE_MAX_ANGLE, BRAKE_MAX_ANGLE)
        
    # print("left_dist =", left_dist, "right_dist =", right_dist)
    # Αν είμαστε κοντά σε τοίχο λιγότερο από SAFETY_DISTANCE
    # μπες σε safety mode ανάλογα με τον πιο κοντινό τοίχο
    if left_dist < SAFETY_DISTANCE or right_dist < SAFETY_DISTANCE:
        if left_dist < right_dist:
            cur_cone_mode = Cone_Mode.left_safety
        else:
            cur_cone_mode = Cone_Mode.right_safety

    # LEFT SAFETY: Είμαστε πολύ κοντά στο να χτυπήσουμε εμπόδιο αριστερά
    # άρα στρίψε ελαφρώς δεξιά
    if cur_cone_mode == Cone_Mode.left_safety:
        angle = 0.2
        speed = SAFETY_SPEED
        if left_dist > END_SAFETY_DISTANCE:
            cur_cone_mode = Cone_Mode.align

    # RIGHT SAFETY: Είμαστε πολύ κοντά στο να χτυπήσουμε εμπόδιο αριστερά
    # άρα στρίψε ελαφρώς αριστερά
    elif cur_cone_mode == Cone_Mode.right_safety:
        angle = -0.2
        speed = SAFETY_SPEED
        if right_dist > END_SAFETY_DISTANCE:
            cur_cone_mode = Cone_Mode.align
    # Προβάλουμε την κροπαρισμένη εικόνα
    rc.display.show_color_image(cropped_image)    

# Lane following που ουσιαστικά είναι line following με εξτρα βήματα
def follow_lane(max_speed = 0.25, sensitivity = 1):
    rc.drive.set_max_speed(max_speed)
    global speed
    global angle

    color_image = rc.camera.get_color_image()
    if color_image is None:
        # Έγινε λάθος και δεν βρήκαμε εικόνα, σπάνιο να γίνει και αν ναι
        # τότε θα γίνει στιγμιαία
        return

    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR_2[0], CROP_FLOOR_2[1])
    # Ψάξε για το primary και το secondary color και βάλε όλα τα περιγράμματα σε
    # μια λίστα
    contours_secondary = [
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

    # Αν δούμε δευτερεύον χρώμα πλησιάζουμε επικίνδυνη στροφή οπότε πήγαινε αργά
    if len(contours_secondary) > 0:
        speed = LANE_FOLLOW_SLOW_SPEED
    else:
        speed = LANE_FOLLOW_FAST_SPEED

    # Εδώ έχουμε τουλάχιστον 2 secondary περιγράμματα, άρα lane following
    if len(contours_secondary) > 1:
        # Κάνουμε sort όλα τα περιγράμματα για να κρατήσουμε τα 2 μεγαλύτερα
        contours_secondary.sort(key=cv.contourArea, reverse=True)

        # Calculate the midpoint of the two largest contours
        first_center = rc_utils.get_contour_center(contours_secondary[0])
        second_center = rc_utils.get_contour_center(contours_secondary[1])
        # Υπολογίζουμε τη μέση (στον άξονα των x) ανάμεσα στα 2 μεγαλύτερα
        # περιγράμματα
        midpoint = (first_center[1] + second_center[1]) / 2

        # Proportional control πάνω στο μέσο σημείο
        angle = rc_utils.remap_range(midpoint, rc.camera.get_width()*4//10, rc.camera.get_width()*6//10, -sensitivity, sensitivity, True)

        # Draw the contours and centers onto the image (red one is larger)
        rc_utils.draw_contour(cropped_image, contours_secondary[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(cropped_image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(cropped_image, contours_secondary[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(cropped_image, second_center, rc_utils.ColorBGR.blue.value)
    # Εδώ έχουμε τουλάχιστον 2 primary περιγράμματα, άρα lane following
    elif len(contours_primary) > 1:
        # Κάνουμε sort όλα τα περιγράμματα για να κρατήσουμε τα 2 μεγαλύτερα
        contours_primary.sort(key=cv.contourArea, reverse=True)

        first_center = rc_utils.get_contour_center(contours_primary[0])
        second_center = rc_utils.get_contour_center(contours_primary[1])
        # Υπολογίζουμε τη μέση (στον άξονα των x) ανάμεσα στα 2 μεγαλύτερα
        # περιγράμματα
        midpoint = (first_center[1] + second_center[1]) / 2

        # Proportional control πάνω στο μέσο σημείο
        angle = rc_utils.remap_range(midpoint, rc.camera.get_width()//4, rc.camera.get_width()*3//4, -sensitivity, sensitivity, True)
        
        # Draw the contours and centers onto the image (red one is larger)
        rc_utils.draw_contour(cropped_image, contours_primary[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(cropped_image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(cropped_image, contours_primary[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(cropped_image, second_center, rc_utils.ColorBGR.blue.value) 

    # Εδώ έχουμε μόνο 1 κύριο περίγραμμα, πηγαίνουμε προς το κέντρο του
    elif len(contours_primary) == 1:
        contour = contours_primary[0]
        center = rc_utils.get_contour_center(contour)
        angle = rc_utils.remap_range(center[1], rc.camera.get_width()//4, rc.camera.get_width()*3//4, -1, 1, True)
        rc_utils.draw_contour(cropped_image, contours_primary[0])
        rc_utils.draw_circle(cropped_image, center)

    # Εδώ έχουμε μόνο 1 δευτερέον περίγραμμα, πηγαίνουμε προς το κέντρο του
    elif len(contours_secondary) == 1:
        contour = contours_secondary[0]
        center = rc_utils.get_contour_center(contour)
        angle = rc_utils.remap_range(center[1], rc.camera.get_width()//4, rc.camera.get_width()*3//4, -1, 1, True)
        rc_utils.draw_contour(cropped_image, contours_secondary[0])
        rc_utils.draw_circle(cropped_image, center)
  
    # Αν δεν βρίσκουμε τίποτα, τότε απλά φώναξε τη follow_lines(), αν και
    # δεν πρόκειται να γίνει ποτέ
    elif len(contours_primary) == 0 and len(contours_secondary) == 0:
        follow_lines(0.25)
        return
    # Προβάλουμε την εικόνα στην οθόνη
    rc.display.show_color_image(cropped_image)

# Wall following για το phase 2
def follow_walls(max_speed = 0.25, sensitivity = 1):
    global follow_walls_counter
    global cur_wall_mode
    global speed
    global angle
    global cur_state
    rc.drive.set_max_speed(max_speed)
    
    follow_walls_counter += rc.get_delta_time()
    if follow_walls_counter < FOLLOW_WALLS_START_LENGTH:
        # Στην αρχή πηγαίνουμε μόνο ευθεία
        speed = 1
        angle = 0
        return
    
    scan = rc.lidar.get_samples()
    speed = 0
    angle = 0

    color_image = rc.camera.get_color_image()
    if color_image is None:
        # Έγινε λάθος και δεν βρήκαμε εικόνα, σπάνιο να γίνει και αν ναι
        # τότε θα γίνει στιγμιαία
        print("No image")
        return

    # Find the minimum distance to the front, side, and rear of the car
    front_angle, front_dist = rc_utils.get_lidar_closest_point(
        scan, (-MIN_SIDE_ANGLE, MIN_SIDE_ANGLE)
    )
    left_angle, left_dist = rc_utils.get_lidar_closest_point(
        scan, (-MAX_SIDE_ANGLE, -MIN_SIDE_ANGLE)
    )
    right_angle, right_dist = rc_utils.get_lidar_closest_point(
        scan, (MIN_SIDE_ANGLE, MAX_SIDE_ANGLE)
    )

    # Estimate the left wall angle relative to the car by comparing the distance
    # to the left-front and left-back
    left_front_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    left_back_dist = rc_utils.get_lidar_average_distance(
        scan, -SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    left_dif = left_front_dist - left_back_dist

    # Use the same process for the right wall angle
    right_front_dist = rc_utils.get_lidar_average_distance(
        scan, SIDE_FRONT_ANGLE, WINDOW_ANGLE
    )
    right_back_dist = rc_utils.get_lidar_average_distance(
        scan, SIDE_BACK_ANGLE, WINDOW_ANGLE
    )
    right_dif = right_front_dist - right_back_dist

    # If we are within SAFETY_DISTANCE of either wall, enter safety mode
    # Αν η αριστερή απόσταση (left_dist) ή η δεξιά απόσταση είναι μικρότερες
    # από το SAFETY_DISTANCE
    if right_dist < SAFETY_DISTANCE  or left_dist < SAFETY_DISTANCE :
        # Αν είμαστε πιο κοντά στον αριστερό τοίχο, μπαίνουμε στο left_safety wall mode
        if right_dist < left_dist:
            cur_wall_mode = Wall_Mode.right_safety
        else:
            cur_wall_mode = Wall_Mode.left_safety
    # If there are no visible walls to follow, stop the car
    # Αν δεν βλέπουμε τοίχο ούτε αριστερά (left_front_dist) ούτε δεξιά
    # (right_front_dist), πήγαινε πολύ αργά ευθεία
    if left_front_dist == 0.0 and right_front_dist == 0.0:
        speed = SAFETY_SPEED
        angle = 0
    # Αν δεν βλέπουμε τοίχο μόνο αριστερά, πήγαινε αργά φουλ αριστερά
    # (γιατί από εκεί είναι ο τοίχος)
    elif left_front_dist == 0.0:
        speed = SAFETY_SPEED
        angle = -1
    # Αν δεν βλέπουμε τοίχο μόνο δεξιά, πήγαινε αργά φουλ δεξιά
    # (γιατί από εκεί είναι ο τοίχος)
    elif right_front_dist == 0.0:
        speed = SAFETY_SPEED
        angle = 1
    # LEFT SAFETY: We are close to hitting a wall to the left, so turn hard right
    # Αν είμαστε στο left_safety wall mode, τότε πήγαινε αργά φουλ δεξιά
    elif cur_wall_mode == Wall_Mode.left_safety:
        speed = SAFETY_SPEED
        angle = 1
        # Αν το left_dist ξεπεράσει το END_SAFETY_DISTANCE, τότε επίστρεψε
        # στο Wall_Mode.align
        if left_dist > END_SAFETY_DISTANCE:
            cur_wall_mode = Wall_Mode.align

    # RIGHT SAFETY: We are close to hitting a wall to the right, so turn hard left
    # Αν είμαστε στο right_safety wall mode, τότε πήγαινε αργά φουλ αριστερά
    elif cur_wall_mode == Wall_Mode.right_safety:
        speed = SAFETY_SPEED
        angle = -1 

        # Αν το right_dist ξεπεράσει το END_SAFETY_DISTANCE, τότε επίστρεψε
        # στο Wall_Mode.align
        if left_dist > END_SAFETY_DISTANCE:
            cur_wall_mode = Wall_Mode.align
            
    # ALIGN: Try to align straight and equidistant between the left and right walls
    else:
        # If left_dif is very large, the hallway likely turns to the left
        #print("left_dif = ", left_dif, "right dif =", right_dif)
        # print("left_dist = ", left_dist, "right dist =", right_dist)
        if left_dif > TURN_THRESHOLD:
            angle = -1

        # Similarly, if right_dif is very large, the hallway likely turns to the right
        elif right_dif > TURN_THRESHOLD:
            angle = 1

        # Otherwise, determine angle by taking into account both the relative angles and
        # distances of the left and right walls
        value = (right_dif - left_dif) + (right_dist - left_dist)
        angle = rc_utils.remap_range(
            value, -TURN_THRESHOLD, TURN_THRESHOLD, -sensitivity, sensitivity, True
        )
        # Choose speed based on the distance of the object in front of the car
        ### Αλλάζουμε το speed με proportional control
        ### Ανάλογα με τη μεταβλητή front_dist, από 0 έως και max_speed
        ### Αν βλέπουμε κάτι μρποστά πηγαίνουμε με 0
        ### Αν κάτι είναι πιο μακρινό από BRAKE_DISTANCE πηγαίνουμε με max_speed
        speed = rc_utils.remap_range(front_dist, 0, BRAKE_DISTANCE, 0, max_speed, True)

    rc.drive.set_speed_angle(speed, angle)    

# Για το phase 7
def pass_trains(max_speed = 0.25):
    global follow_walls_counter
    global speed
    global angle
    global cur_state
    rc.drive.set_max_speed(max_speed)

    scan = rc.lidar.get_samples()
    speed = 0
    angle = 0

    color_image = rc.camera.get_color_image()
    if color_image is None:
        # Έγινε λάθος και δεν βρήκαμε εικόνα, σπάνιο να γίνει και αν ναι
        # τότε θα γίνει στιγμιαία
        print("No image")
        return

    # Find the minimum distance to the front, side, and rear of the car
    front_angle, front_dist = rc_utils.get_lidar_closest_point(
        scan, (-MIN_SIDE_ANGLE, MIN_SIDE_ANGLE)
    )
    left_angle, left_dist = rc_utils.get_lidar_closest_point(
        scan, (-MAX_SIDE_ANGLE, -MIN_SIDE_ANGLE)
    )
    right_angle, right_dist = rc_utils.get_lidar_closest_point(
        scan, (MIN_SIDE_ANGLE, MAX_SIDE_ANGLE)
    )
    print(front_dist)
    speed = rc_utils.remap_range(front_dist, 100, 200, 0, 1, True)
    
# Κοιτάμε μπροστά μας και αν έχουμε χτυπήσει κάπου μπαίνουμε
# στο ειδκό State.recovery. Πριν μπούμε αποθηκεύουμε το
# προηγούμενο state στο save_state
def check_surroundings():
    global cur_state
    global save_state
    global recovery_counter
    scan = rc.lidar.get_samples()
    # Find the minimum distance to the front of the car
    front_angle, front_dist = rc_utils.get_lidar_closest_point(
        scan, (-MIN_SIDE_ANGLE, MIN_SIDE_ANGLE)
    )
    #print(front_dist)
    if cur_state != State.recover and front_dist < 25:
        save_state = cur_state
        cur_state = State.recover
        recovery_counter = 0

# Προσπαθούμε να επαναφερθούμε σε κάποια φυσιολογική κατάσταση
def recover():
    global cur_state
    global save_state
    global recovery_counter
    global recovery_time
    global speed
    global angle
    rc.drive.set_max_speed(0.25)
    speed = RECOVERY_SPEED
    angle = RECOVERY_ANGLE

    recovery_counter += rc.get_delta_time()
    if recovery_counter > recovery_time:
        # Επαναφέρουμε το προηγούμενο state
        cur_state = save_state
        recovery_time += 1


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
