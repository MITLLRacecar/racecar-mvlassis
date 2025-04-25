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

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar() 

# Κατάσταση που ορίζει τη φάση στην οποία βρισκόμαστε
class State(enum.IntEnum):
    start = 0
    line_following = 1
    lane_following = 2
    cone_slaloming = 3
    wall_following = 4
    wall_following_2 = 5
    done = 6

class Wall_Mode(enum.IntEnum):
    align = 0
    right_safety = 1
    left_safety = 2
    reverse = 3

class Cone_Mode(enum.IntEnum):
    red_align = 0  # Approaching a red cone to pass
    blue_align = 1  # Approaching a blue cone to pass
    red_pass = 2  # Passing a red cone (currently out of sight to our left)
    blue_pass = 3  # Passing a blue cone (currently out of sight to our right)
    red_find = 4  # Finding a red cone with which to align
    blue_find = 5  # Finding a blue cone with which to align
    red_reverse = 6  # Aligning with a red cone, but we are too close so must back up
    blue_reverse = 7  # Aligning with a blue cone, but we are too close so must back up
    no_cones_blue = 8  # No cones in sight, inch forward until we find one
    no_cones_red = 9

# Τρέχουσα φάση στην οποία βρισκόμαστε
cur_state = State.start
cur_wall_mode = Wall_Mode.align
cur_marker = rc_utils.ARMarker(-1, np.zeros((4, 2), dtype=np.int32)) # Κενό marker

# Minimum number of pixels to consider a valid contour
MIN_CONTOUR_AREA = 30

# Το παράθυρο που θα κρατάμε κάθε φορά μετά το crop, ορίζεται από το πάνω
# αριστερά και το κάτω δεξιά σημείο. Κάθε σημείο είναι της μορφής (row, column)
CROP_FLOOR = (
    (rc.camera.get_height() * 2 // 3, 0),
    (rc.camera.get_height(), rc.camera.get_width())
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
LINE_FOLLOW_SPEED = 1 # Ταχύτητα με την οποία ακολουθούμε τις γραμμές
MAX_LINE_DISTANCE = 100 # Μέγιστη απόσταση από την οποία θα ακολουθούμε μια γραμμή
# Προτεραιότητες χρωμάτων που θα ακολουθήσουμε στη φάση 1
# Είναι πίνακας με 5 χρώματα, αρχικά θέλουμε να είναι μια καθορισμένη ακολουθία
# χρωμάτων, αλλά θα την αλλάξουμε αφού διαβάσουμε τα AR Markers
colors = [PURPLE, ORANGE, RED, GREEN, BLUE]

# For phase 2
CROP_FLOOR_2 = (
    (rc.camera.get_height() * 3 // 5, 0),
    (rc.camera.get_height(), rc.camera.get_width())
) 
primary_color = ORANGE # Χρώμα που ορίζει τις ευθείες
secondary_color = PURPLE # Χρώμα που ορίζει τις απότομες στροφές
# Speed constants
LANE_FOLLOW_FAST_SPEED = 0.9
LANE_FOLLOW_SLOW_SPEED = 0.25
# Angle constants
# Amount to turn if we only see one lane
ONE_LANE_TURN_ANGLE = 1
MIN_LANE_CONTOUR_AREA = 40
HARD_TURN = False # True if we are currently making a hard turn
lane_hard_turn_counter = 0
counter = 0

# For phase 3
CONE_SLALOM_SPEED = 1
MAX_ALIGN_SPEED = 0.8
MIN_ALIGN_SPEED = 0.4
PASS_SPEED = 0.5
FIND_SPEED = 0.2
REVERSE_SPEED = -0.2
NO_CONES_SPEED = 1
REVERSE_BRAKE_TIME = 0.25
SHORT_PASS_TIME = 1.0
LONG_PASS_TIME = 1.2
CROP_FLOOR_3 = (
    (rc.camera.get_height() * 3 // 10, 0),
    (rc.camera.get_height(), rc.camera.get_width())
)
MIN_CONE_AREA = 100
MAX_DISTANCE = 250
REVERSE_DISTANCE = 50
STOP_REVERSE_DISTANCE = 60
CLOSE_DISTANCE = 30
FAR_DISTANCE = 120
cur_cone_mode = Cone_Mode.no_cones_blue
red_center = None
red_distance = 0
prev_red_distance = 0
blue_center = None
blue_distance = 0
prev_blue_distance = 0
phase4_flag = False

# For phase 4
# >> Constants
# The maximum speed the car will travel
MAX_SPEED = 0.70
# When an object in front of the car is closer than this (in cm), start braking
BRAKE_DISTANCE = 150
# When a wall is within this distance (in cm), focus solely on not hitting that wall
safety_DISTANCE = 30
# When a wall is greater than this distance (in cm) away, exit safety mode
END_safety_DISTANCE = 32
# Speed to travel in safety mode
safety_SPEED = 0.2
# The minimum and maximum angles to consider when measuring closest side distance
MIN_SIDE_ANGLE = 10
MAX_SIDE_ANGLE = 60
# The angles of the two distance measurements used to estimate the angle of the left
# and right walls
SIDE_FRONT_ANGLE = 70
SIDE_BACK_ANGLE = 110
# When the front and back distance measurements of a wall differ by more than this
# amount (in cm), assume that the hallway turns and begin turning
TURN_THRESHOLD = 20
# The angle of measurements to average when taking an average distance measurement
WINDOW_ANGLE = 12
HARD_TURN_FLAG = False
HARD_TURN_WALL = "RIGHT"
hard_turn_counter = 0

def start():
    """
    This function is run once every time the start button is pressed
    """
    global cur_state
    global counter
    global speed
    global angle

    # Ξεκινάμε ακίνητοι
    rc.drive.stop()
    
    # Η παρακάτω συνάρτηση αυξάνει τη ΜΕΓΙΣΤΗ ταχύτητα του αυτοκινήτου
    # Πηγαίνει από 0 μέχρι 1, by default είναι 0.25
    # Την αλλάζουμε με πολλή προσοχή
    rc.drive.set_max_speed(0.3)
    
    # Ξεκινάμε στο αρχικό mode μέχρι να εντοπίσουμε AR Marker
    cur_state = State.start
    counter = 0
    speed = 0
    angle = 0

    # Print start message
    print(">> Time Trial Challenge - Jumpscar - Charalambos Kokkinos, Dimitris Panagiotakopoulos, Vasilis Petropoulos, Konstantinos Stavrou")

def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle
    global cur_state
    global cur_marker
    global first_color
    global second_color

    first_color = "RED"
    second_color = "GREEN"

    change_state()
    if cur_state == State.line_following:
        follow_lines(max_speed=0.60)
        ang_vel = rc.physics.get_angular_velocity()
        if ang_vel[2] < -0.04:
            speed = -0.5
    elif cur_state == State.lane_following:
        follow_lane(0.75)
    elif cur_state == State.cone_slaloming:
        cone_slalom(0.38)
        #cone_slalom2(0.25)
    elif cur_state == State.wall_following:
        follow_walls(0.23, 1)
    elif cur_state == State.wall_following_2:
        follow_walls(0.8, 0.4)
    
    rc.drive.set_speed_angle(speed, angle)


# Καλείται κάθε 1 δευτερόλεπτο, χρησιμοποιείται για debugging (αποσφαλμάτωση)
def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    print("Current state:", cur_state)
    if cur_state == State.wall_following:
        print("Current wall_mode:", cur_wall_mode)


# Μετάβαση στην επόμενη φάση και απαραίτητες ενέργειες για να προετοιμαστούμε
def change_state():
    global cur_state
    global colors
    global first_color
    global second_color
    global primary_color
    global secondary_color
    global phase4_flag

    # Εδώ παίρνουμε όλα τα markers που βλέπουμε
    color_image = rc.camera.get_color_image_no_copy()
    markers = rc_utils.get_ar_markers(color_image)

    # for marker in markers:
    #     print(str(marker))
    #     pass

    if cur_state == State.start:
        # Βρίσκουμε του marker που δείχνει αριστερά
        # και βάζουμε το χρώμα του πρώτο στην προτεραιότητα
        for marker in markers:
            # Παίρνουμε την κατεύθυνση του marker
            if marker.get_orientation() == rc_utils.Orientation.LEFT:
                # Εδώ κοιτάμε το αριστερό marker, θέλουμε να είναι το πρώτο χρώμα
                first_color = marker.get_color() # Το χρώμα του marker
                for color in colors:
                    if first_color == color[2]: # color[2] = το όνομα του χρώματος
                        colors.remove(color) # Βγάλε το χρώμα
                        colors.insert(0, color) # Και βάλε το στην αρχή
            elif marker.get_orientation() == rc_utils.Orientation.RIGHT:
                # Εδώ κοιτάμε το δεξί marker, το χρειαζόμαστε για μετά
                second_color = marker.get_color()
        cur_state = State.line_following
        
    elif cur_state == State.line_following:
        for marker in markers:
            corners = marker.get_corners()
            marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
            # Τραβάμε φωτογραφία βάθους
            depth_image = rc.camera.get_depth_image()
            # και βρίσκουμε την απόσταση του marker από εμάς
            marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
            # Αν είμαστε κοντά στο marker
            if int(marker.get_id()) == 1 and 0 < marker_distance < 100:
                cur_state = State.cone_slaloming
                # Επιστρέφουμε την προτεραιότητα στο κανονικό
                colors = [PURPLE, ORANGE, RED, GREEN, BLUE]
                colors_copy = [PURPLE, ORANGE, RED, GREEN, BLUE]
                # Βάζουμε πρώτα στην προτεραίτητα το χρώμα που δεν υπήρχε στα AR markers
                for color in colors:
                    if first_color == color[2]:
                        # Αφαιρούμε το χρώμα με τη remove() από τη λίστα και το βάζουμε
                        # στο τέλος με την append()
                        colors_copy.remove(color)
                        colors_copy.append(color)
                    elif second_color == color[2]:
                        # Αφαιρούμε το χρώμα με τη remove() από τη λίστα και το βάζουμε
                        # στο τέλος με την append()
                        colors_copy.remove(color)
                        colors_copy.append(color)
                cur_state = State.lane_following
                colors = colors_copy
        
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
            elif int(marker.get_id()) == 2 and marker.get_orientation() == rc_utils.Orientation.UP:
                # Αν βρούμε το επόμενο marker και είναι αρκετά κοντά
                # αλλάζουμε κατάσταση
                corners = marker.get_corners()
                marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
                # Τραβάμε φωτογραφία βάθους
                depth_image = rc.camera.get_depth_image()
                # και βρίσκουμε την απόσταση του marker από εμάς
                marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
                # Αν η απόσταση του marker από εμάς είναι μικρότερη από 100 εκατοστά
                if 0 < marker_distance < 200:
                    cur_state = State.cone_slaloming
    elif cur_state == State.cone_slaloming:
        for marker in markers:
            # Αν βρούμε το επόμενο marker αλλάζουμε κατάσταση
            if int(marker.get_id()) == 3 and marker.get_orientation() == rc_utils.Orientation.UP:
                corners = marker.get_corners()
                marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
                # Τραβάμε φωτογραφία βάθους
                depth_image = rc.camera.get_depth_image()
                # και βρίσκουμε την απόσταση του marker από εμάς
                marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
                # Αν η απόσταση του marker από εμάς είναι μικρότερη από 100 εκατοστά
                if 0 < marker_distance < 100:
                    cur_state = State.wall_following
                    cur_wall_mode = Wall_Mode.align
                if 100 < marker_distance < 200:
                    phase4_flag = True

    elif cur_state == State.wall_following:
        pass


# Για τη φάση 1: Line following όπως στο Lab 2A
def follow_lines(max_speed=0.25, sensitivity=1):
    rc.drive.set_max_speed(max_speed)
    global cur_mode
    global speed
    global angle

    speed = LINE_FOLLOW_SPEED
    
    color_image = rc.camera.get_color_image()
    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])
    no_colors_found = True

    # Depth image για να αγνοούμε κάποια γραμμή αν είναι πολύ μακριά
    depth_image = rc.camera.get_depth_image()
    cropped_depth_image = rc_utils.crop(depth_image, CROP_FLOOR[0], CROP_FLOOR[1])

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
            no_colors_found = False
            center = rc_utils.get_contour_center(contour)
            contour_distance = rc_utils.get_pixel_average_distance(cropped_depth_image,
                                                                   center)
            # Αν η απόσταση της γραμμής είναι πολύ μεγάλη, τότε αγνόησε την
            if (contour_distance > MAX_LINE_DISTANCE):
                no_colors_found = True
                continue
            
            # Το remap_range είναι πιο απότομο από ότι συνήθως
            # από το 1/4 της οθόνης και μετά στρίβουμε πλήρως
            angle = rc_utils.remap_range(center[1], rc.camera.get_width()*1//4, rc.camera.get_width()*3//4, -sensitivity, sensitivity, True)
            rc_utils.draw_contour(cropped_image, contour)
            rc_utils.draw_circle(cropped_image, center)
            break
    rc.display.show_color_image(cropped_image)

# Για τη φάση 2: Lane following που ουσιαστικά είναι line following με εξτρα βήματα
def follow_lane(max_speed=0.25):
    rc.drive.set_max_speed(max_speed)
    global speed
    global angle
    global HARD_TURN
    global hard_turn_counter

    color_image = rc.camera.get_color_image()
    if color_image is None:
        # Έγινε λάθος και δεν βρήκαμε εικόνα, σπάνιο να γίνει και αν ναι
        # τότε θα γίνει στιγμιαία
        print("No image")
        return
    markers = rc_utils.get_ar_markers(color_image)

    # Διαβάζουμε το marker που δείχνει τη σωστή στροφή στη δεύτερη φάση
    if len(markers) == 1 and int(markers[0].get_id()) == 199:
        # Παίρνουμε τις συντεταγμένες των 4 γωνιών του τετραγώνου
        corners = markers[0].get_corners()
        # Και με βάση αυτές βρίσκουμε το κέντρο του πάνω στην εικόνα
        marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
        # Τραβάμε φωτογραφία βάθους
        depth_image = rc.camera.get_depth_image()
        # και βρίσκουμε την απόσταση του από εμάς
        marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
        # print("marker_distance is: ", marker_distance)
        # Αν η απόσταση του marker από εμάς είναι μικρότερη από 100 εκατοστά
        if 0 < marker_distance < 200:
            print("Orientation!")
            # Στρίβουμε δεξιά αν το orientation marker του είναι προς τα δεξιά
            if markers[0].get_orientation() == rc_utils.Orientation.LEFT:
                angle = -1
            # Αλλιώς στρίβουμε αριστερά
            else:
                angle = 1
            cropped_image = rc_utils.crop(color_image, CROP_FLOOR_2[0], CROP_FLOOR_2[1])
            rc.display.show_color_image(cropped_image)
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

    speed = LANE_FOLLOW_FAST_SPEED

    # Αν δεν βρίσκουμε τίποτα, τότε απλά ψάξε για τα άλλα 3 χρώματα από το phase 1
    if len(contours_primary) == 0 and len(contours_secondary) == 0:
        follow_lines()
        return
    
    elif HARD_TURN == True or (len(contours_secondary) <= 1 and len(contours_primary) <= 1):
        # Εδώ είμαστε μέσα στο HARD_TURN. Στρίβουμε συνέχεια φουλ αριστερά
        # ή φουλ δεξιά μέχρι να βρούμε τη λωρίδα του primary color
        if not HARD_TURN:
            if len(contours_secondary) > 0:
                contour = contours_secondary[0]
            elif len(contours_primary) > 0:
                contour = contours_primary[0]
            else:
                contour = None
            center = rc_utils.get_contour_center(contour)
            print("HARD TURN STARTED")
            if center[1] > rc.camera.get_width() // 2:
                angle = -1
            else:
                angle = 1
            HARD_TURN = True
            rc_utils.draw_contour(cropped_image, contour)
            rc_utils.draw_circle(cropped_image, center)
        # Αν βρούμε τουλάχιστον 2 περιγράμματα του primary color, τότε
        # σταματάμε το hard turn
        if len(contours_primary) > 1 or len(contours_secondary) > 1:
            HARD_TURN = False

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
        angle = rc_utils.remap_range(midpoint, rc.camera.get_width()//4, rc.camera.get_width()*3//4, -1, 1, True)
        
        # Draw the contours and centers onto the image (red one is larger)
        rc_utils.draw_contour(cropped_image, contours_primary[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(cropped_image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(cropped_image, contours_primary[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(cropped_image, second_center, rc_utils.ColorBGR.blue.value)

    # Εδώ έχουμε τουλάχιστον 2 secondary περιγράμματα, άρα lane following
    elif len(contours_secondary) > 1:
        # Κάνουμε sort όλα τα περιγράμματα για να κρατήσουμε τα 2 μεγαλύτερα
        contours_secondary.sort(key=cv.contourArea, reverse=True)

        # Calculate the midpoint of the two largest contours
        first_center = rc_utils.get_contour_center(contours_secondary[0])
        second_center = rc_utils.get_contour_center(contours_secondary[1])
        # Υπολογίζουμε τη μέση (στον άξονα των x) ανάμεσα στα 2 μεγαλύτερα
        # περιγράμματα
        midpoint = (first_center[1] + second_center[1]) / 2

        # Proportional control πάνω στο μέσο σημείο
        angle = rc_utils.remap_range(midpoint, rc.camera.get_width()*4//10, rc.camera.get_width()*6//10, -1, 1, True)

        # Draw the contours and centers onto the image (red one is larger)
        rc_utils.draw_contour(cropped_image, contours_secondary[0], rc_utils.ColorBGR.red.value)
        rc_utils.draw_circle(cropped_image, first_center, rc_utils.ColorBGR.red.value)
        rc_utils.draw_contour(cropped_image, contours_secondary[1], rc_utils.ColorBGR.blue.value)
        rc_utils.draw_circle(cropped_image, second_center, rc_utils.ColorBGR.blue.value)

    # Εδώ έχουμε μόνο 1 κύριο περίγραμμα, πηγαίνουμε προς το κέντρο του
    elif len(contours_primary) == 1:
        contour = contours_primary[0]
        center = rc_utils.get_contour_center(contour)
        angle = rc_utils.remap_range(center[1], rc.camera.get_width()//4, rc.camera.get_width()*3//4, -1, 1, True)
        rc_utils.draw_contour(cropped_image, contours_primary[0])
        rc_utils.draw_circle(cropped_image, center)
    # Display the image to the screen
    rc.display.show_color_image(cropped_image)

# Για να βρίσκουμε τους κώνους    
def find_cones():
    """
    Find the closest red and blue cones and update corresponding global variables.
    """
    global red_center
    global red_distance
    global prev_red_distance
    global blue_center
    global blue_distance
    global prev_blue_distance

    prev_red_distance = red_distance
    prev_blue_distance = blue_distance

    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()

    if color_image is None or depth_image is None:
        red_center = None
        red_distance = 0
        blue_center = None
        blue_distance = 0
        print("No image found")
        return

    # Search for the red cone
    contours = rc_utils.find_contours(color_image, RED[0], RED[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        red_center = rc_utils.get_contour_center(contour)
        red_distance = rc_utils.get_pixel_average_distance(depth_image, red_center)

        # Only use count it if the cone is less than MAX_DISTANCE away
        if red_distance <= MAX_DISTANCE:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.green.value)
            rc_utils.draw_circle(color_image, red_center, rc_utils.ColorBGR.green.value)
        else:
            red_center = None
            red_distance = 0
    else:
        red_center = None
        red_distance = 0

    # Search for the blue cone
    contours = rc_utils.find_contours(color_image, BLUE[0], BLUE[1])
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if contour is not None:
        blue_center = rc_utils.get_contour_center(contour)
        blue_distance = rc_utils.get_pixel_average_distance(depth_image, blue_center)

        # Only use count it if the cone is less than MAX_DISTANCE away
        if blue_distance <= MAX_DISTANCE:
            rc_utils.draw_contour(color_image, contour, rc_utils.ColorBGR.yellow.value)
            rc_utils.draw_circle(
                color_image, blue_center, rc_utils.ColorBGR.yellow.value
            )
        else:
            blue_center = None
            blue_distance = 0
    else:
        blue_center = None
        blue_distance = 0

    rc.display.show_color_image(color_image)

def cone_slalom2(max_speed=0.25):
    global speed
    global angle
    global cur_cone_mode
    global counter
    rc.drive.set_max_speed(max_speed)

    find_cones()
    print(cur_cone_mode)
    # print(counter)

    # Align ourselves to smoothly approach and pass the red cone while it is in view
    if cur_cone_mode == Cone_Mode.red_align:
        # Once the red cone is out of view, enter Cone_Mode.red_pass
        if (
            red_center is None
            or red_distance == 0
            or red_distance - prev_red_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_red_distance < FAR_DISTANCE:
                counter = max(SHORT_PASS_TIME, counter)
                cur_cone_mode = Cone_Mode.red_pass
            else:
                cur_cone_mode = Cone_Mode.no_cones_blue

        # If it seems like we are not going to make the turn, enter Cone_Mode.red_reverse
        elif (
            red_distance < REVERSE_DISTANCE
            and red_center[1] > rc.camera.get_width() // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_cone_mode = Cone_Mode.red_reverse

        # Align with the cone so that it gets closer to the left side of the screen
        # as we get closer to it, and slow down as we approach
        else:
            goal_point = rc_utils.remap_range(
                red_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                0,
                rc.camera.get_width() // 4,
                True,
            )

            angle = rc_utils.remap_range(
                red_center[1], goal_point, rc.camera.get_width() // 2, 0, 1
            )
            angle = rc_utils.clamp(angle, -1, 1)

            speed = rc_utils.remap_range(
                red_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                MIN_ALIGN_SPEED,
                MAX_ALIGN_SPEED,
                True,
            )

    elif cur_cone_mode == Cone_Mode.blue_align:
        if (
            blue_center is None
            or blue_distance == 0
            or blue_distance - prev_blue_distance > CLOSE_DISTANCE
        ):
            if 0 < prev_blue_distance < FAR_DISTANCE:
                counter = max(SHORT_PASS_TIME, counter)
                cur_cone_mode = Cone_Mode.blue_pass
            else:
                cur_cone_mode = Cone_Mode.no_cones_red
        elif (
            blue_distance < REVERSE_DISTANCE
            and blue_center[1] < rc.camera.get_width() * 3 // 4
        ):
            counter = REVERSE_BRAKE_TIME
            cur_cone_mode = Cone_Mode.blue_reverse
        else:
            goal_point = rc_utils.remap_range(
                blue_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                rc.camera.get_width(),
                rc.camera.get_width() * 3 // 4,
                True,
            )

            angle = rc_utils.remap_range(
                blue_center[1], goal_point, rc.camera.get_width() // 2, 0, -1
            )
            angle = rc_utils.clamp(angle, -1, 1)

            speed = rc_utils.remap_range(
                blue_distance,
                CLOSE_DISTANCE,
                FAR_DISTANCE,
                MIN_ALIGN_SPEED,
                MAX_ALIGN_SPEED,
                True,
            )

    # Curve around the cone at a fixed speed for a fixed time to pass it
    if cur_cone_mode == Cone_Mode.red_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, -0.5, True)
        speed = PASS_SPEED
        counter -= rc.get_delta_time()
        # After the counter expires, enter Cone_Mode.blue_align if we see the blue cone,
        # and Cone_Mode.blue_find if we do not
        if counter <= 0:
            if 0 < blue_distance < red_distance:
                cur_cone_mode = Cone_Mode.blue_align if blue_distance > 0 else Cone_Mode.blue_find
            elif 0 < red_distance < blue_distance:
                cur_cone_mode = Cone_Mode.red_align if red_distance > 0 else Cone_Mode.red_find
            else:
                cur_cone_mode = Cone_Mode.no_cones_blue
                

    elif cur_cone_mode == Cone_Mode.blue_pass:
        angle = rc_utils.remap_range(counter, 1, 0, 0, 0.5, True)
        speed = PASS_SPEED
        counter -= rc.get_delta_time()
        if counter <= 0:
            if 0 < blue_distance < red_distance:
                cur_cone_mode = Cone_Mode.blue_align if blue_distance > 0 else Cone_Mode.blue_find
            elif 0 < red_distance < blue_distance:
                cur_cone_mode = Cone_Mode.red_align if red_distance > 0 else Cone_Mode.red_find
            else:
                cur_cone_mode = Cone_Mode.no_cones_red

    # If we know we are supposed to be aligning with a red cone but do not see one,
    # turn to the right until we find it
    elif cur_cone_mode == Cone_Mode.red_find:
        angle = 1
        speed = FIND_SPEED
        if red_distance > 0:
            cur_cone_mode = Cone_Mode.red_align

    elif cur_cone_mode == Cone_Mode.blue_find:
        angle = -1
        speed = FIND_SPEED
        if blue_distance > 0:
            cur_cone_mode = Cone_Mode.blue_align

    # If we are not going to make the turn, reverse while keeping the cone in view
    elif cur_cone_mode == Cone_Mode.red_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = -1
            speed = REVERSE_SPEED
            if (
                red_distance > STOP_REVERSE_DISTANCE
                or red_center is not None and red_center[1] < rc.camera.get_width() // 10
            ):
                counter = LONG_PASS_TIME
                cur_cone_mode = Cone_Mode.red_align

    elif cur_cone_mode == Cone_Mode.blue_reverse:
        if counter >= 0:
            counter -= rc.get_delta_time()
            speed = -1
            angle = 1
        else:
            angle = 1
            speed = REVERSE_SPEED
            if (
                blue_distance > STOP_REVERSE_DISTANCE
                or blue_center is not None and blue_center[1] > rc.camera.get_width() * 9 / 10
            ):
                counter = LONG_PASS_TIME
                cur_cone_mode = Cone_Mode.blue_align

    # If no cones are seen, drive forward until we see either a red or blue cone
    elif cur_cone_mode == Cone_Mode.no_cones_blue:
        angle = 0.6
        speed = NO_CONES_SPEED

        if red_distance > 0 and blue_distance == 0:
            cur_cone_mode = Cone_Mode.red_align
        elif blue_distance > 0 and red_distance == 0:
            cur_cone_mode = Cone_Mode.blue_align
        elif blue_distance > 0 and red_distance > 0:
            cur_cone_mode = (
                Cone_Mode.red_align if red_distance < blue_distance else Cone_Mode.blue_align
            )
    elif cur_cone_mode == Cone_Mode.no_cones_red:
        angle = -0.6
        speed = NO_CONES_SPEED

        if red_distance > 0 and blue_distance == 0:
            cur_cone_mode = Cone_Mode.red_align
        elif blue_distance > 0 and red_distance == 0:
            cur_cone_mode = Cone_Mode.blue_align
        elif blue_distance > 0 and red_distance > 0:
            cur_cone_mode = (
                Cone_Mode.red_align if red_distance < blue_distance else Cone_Mode.blue_align
            )

    rc.drive.set_speed_angle(speed, angle)    
    
# Για τη φάση 3: Cone slaloming
def cone_slalom(max_speed=0.25):
    rc.drive.set_max_speed(max_speed)
    global cur_mode
    global speed
    global angle

    speed = CONE_SLALOM_SPEED
    
    color_image = rc.camera.get_color_image()
    # Crop to the floor directly in front of the car
    cropped_image = rc_utils.crop(color_image, CROP_FLOOR_3[0], CROP_FLOOR_3[1])

    contours_blue = [
        contour
        for contour in rc_utils.find_contours(
            cropped_image, BLUE[0], RED[1]
        )
        if cv.contourArea(contour) > MIN_CONE_AREA
    ]
    contours_red = [
        contour
        for contour in rc_utils.find_contours(
            cropped_image, RED[0], RED[1]
        )
        if cv.contourArea(contour) > MIN_CONE_AREA
    ]

    if phase4_flag:
        speed = 0.2
        angle = 0
        rc.display.show_color_image(cropped_image)
        return

    contours_all = contours_blue + contours_red
    contours_all.sort(key=cv.contourArea) # τα κάνουμε sort για να κρατήσουμε το μικρότερο
    if contours_all:
        if len(contours_all) > 1:
            first_center = rc_utils.get_contour_center(contours_all[0])
            second_center = rc_utils.get_contour_center(contours_all[1])
            # Βρίσκουμε το σημείο ανάμεσα στα 2 περιγράμματα
            midpoint = (first_center[1] + second_center[1]) / 2

            # Proportional control πάνω στο μέσο σημείο
            angle = rc_utils.remap_range(midpoint, rc.camera.get_width()//4, rc.camera.get_width()*3//4, -1, 1, True)

            # Draw the contours and centers onto the image (red one is larger)
            rc_utils.draw_contour(cropped_image, contours_all[0], rc_utils.ColorBGR.red.value)
            rc_utils.draw_circle(cropped_image, first_center, rc_utils.ColorBGR.red.value)
            rc_utils.draw_contour(cropped_image, contours_all[1], rc_utils.ColorBGR.blue.value)
            rc_utils.draw_circle(cropped_image, second_center, rc_utils.ColorBGR.blue.value)
        else:
            center = rc_utils.get_contour_center(contours_all[0])
            # Proportional control στο κέντρο του μικρότερου περιγράμματος
            angle = rc_utils.remap_range(center[1], 0, rc.camera.get_width(), -1, 1, True)
            rc_utils.draw_contour(cropped_image, contours_all[0])
            rc_utils.draw_circle(cropped_image, center)
          
    rc.display.show_color_image(cropped_image)

def follow_walls(max_speed, sensitivity):
    rc.drive.set_max_speed(max_speed)
    global cur_wall_mode
    global speed
    global angle
    global cur_state
    global HARD_TURN_WALL
    global HARD_TURN_FLAG
    global hard_turn_counter
    scan = rc.lidar.get_samples()
    speed = 0
    angle = 0

    color_image = rc.camera.get_color_image()
    if color_image is None:
        # Έγινε λάθος και δεν βρήκαμε εικόνα, σπάνιο να γίνει και αν ναι
        # τότε θα γίνει στιγμιαία
        print("No image")
        return
    markers = rc_utils.get_ar_markers(color_image)

    # Διαβάζουμε το marker που δείχνει τη σωστή στροφή στη δεύτερη φάση
    if len(markers) == 1 and int(markers[0].get_id()) == 199:
        # Παίρνουμε τις συντεταγμένες των 4 γωνιών του τετραγώνου
        corners = markers[0].get_corners()
        # Και με βάση αυτές βρίσκουμε το κέντρο του πάνω στην εικόνα
        marker_center = ((corners[0][0] + corners[2][0]) // 2, (corners[0][1] + corners[2][1]) // 2)
        # print(marker_center)
        # Τραβάμε φωτογραφία βάθους
        depth_image = rc.camera.get_depth_image()
        # και βρίσκουμε την απόσταση του από εμάς
        marker_distance = rc_utils.get_pixel_average_distance(depth_image, marker_center)
        #print("marker_distance is", marker_distance)
        # print("marker_distance is: ", marker_distance)
        # Αν η απόσταση του marker από εμάς είναι μικρότερη από 80 εκατοστά
        if 0 < marker_distance < 70:
            speed = 0.3
            if markers[0].get_orientation() == rc_utils.Orientation.RIGHT:
                # Αν το marker δείχνει προς τα δεξιά, το HARD_TURN_WALL γίνεται "RIGHT"
                # και στρίβουμε full δεξιά
                HARD_TURN_WALL = "RIGHT"
                angle = 1
            else:
                # Αλλιώς γίνεται "LEFT" και στρίβουμε φουλ αριστερά
                HARD_TURN_WALL = "LEFT"
                angle = -1
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

    # If we are within safety_DISTANCE of either wall, enter safety mode
    # Αν η αριστερή απόσταση (left_dist) ή η δεξιά απόσταση είναι μικρότερες
    # από το safety_DISTANCE
    if right_dist < safety_DISTANCE  or left_dist < safety_DISTANCE :
        # Αν είμαστε πιο κοντά στον αριστερό τοίχο, μπαίνουμε στο left_safety wall mode
        if right_dist < left_dist:
            cur_wall_mode = Wall_Mode.right_safety
        else:
            cur_wall_mode = Wall_Mode.left_safety
    # If there are no visible walls to follow, stop the car
    # Αν δεν βλέπουμε τοίχο ούτε αριστερά (left_front_dist) ούτε δεξιά
    # (right_front_dist), πήγαινε πολύ αργά ευθεία
    if left_front_dist == 0.0 and right_front_dist == 0.0:
        speed = safety_SPEED
        angle = 0
    # Αν δεν βλέπουμε τοίχο μόνο αριστερά, πήγαινε αργά φουλ αριστερά
    # (γιατί από εκεί είναι ο τοίχος)
    elif left_front_dist == 0.0:
        speed = safety_SPEED
        angle = -1
    # Αν δεν βλέπουμε τοίχο μόνο δεξιά, πήγαινε αργά φουλ δεξιά
    # (γιατί από εκεί είναι ο τοίχος)
    elif right_front_dist == 0.0:
        speed = safety_SPEED
        angle = 1
    # LEFT SAFETY: We are close to hitting a wall to the left, so turn hard right
    # Αν είμαστε στο left_safety wall mode, τότε πήγαινε αργά φουλ δεξιά
    elif cur_wall_mode == Wall_Mode.left_safety:
        speed = safety_SPEED
        angle = 1
        # Αν το left_dist ξεπεράσει το END_SAFETY_DISTANCE, τότε επίστρεψε
        # στο Wall_Mode.align
        if left_dist > END_safety_DISTANCE:
            cur_wall_mode = Wall_Mode.align

    # RIGHT SAFETY: We are close to hitting a wall to the right, so turn hard left
    # Αν είμαστε στο right_safety wall mode, τότε πήγαινε αργά φουλ αριστερά
    elif cur_wall_mode == Wall_Mode.right_safety:
        speed = safety_SPEED
        angle = -1 

        # Αν το right_dist ξεπεράσει το END_SAFETY_DISTANCE, τότε επίστρεψε
        # στο Wall_Mode.align
        if left_dist > END_safety_DISTANCE:
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

        if HARD_TURN_FLAG == True:
            speed = 0.2
            if HARD_TURN_WALL == "LEFT":
                angle = -1
            else:
                angle = 1
            hard_turn_counter += rc.get_delta_time()
            if hard_turn_counter > 2.5:
                HARD_TURN_FLAG = False
                cur_state = State.wall_following_2

        elif ((HARD_TURN_WALL == "LEFT" and left_dif > 94 and right_dif > 94)
        or HARD_TURN_WALL == "RIGHT" and left_dif > 110 and right_dif > 110):
            HARD_TURN_FLAG = True
            hard_turn_counter = 0
            speed = 0.2
            # Στρίβουμε σε κατάλληλη γωνία ανάλογα με την τιμή του HARD_TURN_WALL
            if HARD_TURN_WALL == "LEFT":
                angle = -1
            else:
                angle = 1
        # Choose speed based on the distance of the object in front of the car
        ### Αλλάζυμε το speed με proportional control
        ### Ανάλογα με τη μεταβλητή front_dist, από 0 έως και MAX_SPEED
        ### Αν βλέπουμε κάτι μρποστά πηγαίνουμε με 0
        ### Αν κάτι είναι πιο μακρινό από BRAKE_DISTANCE πηγαίνουμε με MAX_SPEED
        speed = rc_utils.remap_range(front_dist, 0, BRAKE_DISTANCE, 0, MAX_SPEED, True)

    rc.drive.set_speed_angle(speed, angle)
    

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
