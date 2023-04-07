import cv2
from cvzone.ColorModule import ColorFinder

hsv_list = [
    {'hmin': 91, 'smin': 188, 'vmin': 36, 'hmax': 153, 'smax': 255, 'vmax': 255, 'color': 'b1'},
    {'hmin': 163, 'smin': 25, 'vmin': 95, 'hmax': 179, 'smax': 255, 'vmax': 255, 'color': 'ro'},
    {'hmin': 15, 'smin': 100, 'vmin': 100, 'hmax': 37, 'smax': 255, 'vmax': 255, 'color': 'y1'},
    {'hmin': 65, 'smin': 195, 'vmin': 69, 'hmax': 179, 'smax': 219, 'vmax': 99, 'color': 'g1'}]
hsv_cue = {'hmin': 38, 'hmax': 78, 'smin': 1, 'smax': 65, 'vmin': 146, 'vmax': 241, 'color': 'w'}

cap = cv2.VideoCapture('E:\AIspecs\Shot.mp4')
capPred = cv2.VideoWriter('E:\AIspecs\Pool_predict.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))
poc_cor = [(162, 95), (632, 82), (1115, 93),
           (188, 520), (632, 545), (1115, 536)]

# Extracting The Contours Of Cue, Cue Ball  And Coloured Balls
def findContours(img, imgpre=None, minArea=1000, sort=True, filter=0, drawCon=True, c=(255, 0, 0), color_check=0):
    conFound = []
    imgContours = img.copy()
    contours, hierarchy = cv2.findContours(imgpre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    minArea += 1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 180:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == filter or filter == 0:
                if drawCon: cv2.drawContours(imgContours, cnt, -1, c, 3)
                x, y, w, h = cv2.boundingRect(approx)
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                if color_check == 0 and 55 < y < 560 and 130 < x < 1140 and radius > 14:
                    cx, cy = int(cx), int(cy)
                    conFound.append(
                        {"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": (cx, cy), "radius": radius})

                elif color_check == 1 and 55 < y < 500 and 130 < x < 1140 and radius > 14 and radius < 30:
                    cx, cy = int(cx), int(cy)
                    conFound.append(
                        {"cnt": cnt, "area": area, "bbox": [x, y, w, h], "center": (cx, cy), "radius": radius})

    if sort:
        conFound = sorted(conFound, key=lambda x: x["radius"], reverse=False)

    return imgContours, conFound

# Collision Check Between the Cue Ball and Colour ball
def collision_check(img, cueb_x, cueb_y, cuel_x, cuel_y, targ_x, targ_y):
    global col_state
    d_x = cuel_x - cueb_x
    d_y = cuel_y - cueb_y
    if col_state == 0:
        for i in range(1, 100):
            i /= 10
            n_x = int(cueb_x - (d_x * i))
            n_y = int(cueb_y - (d_y * i))
            if ball_touch_check(img, n_x, n_y, targ_x, targ_y):
                draw_points_2.append([cueb_x, cueb_y, n_x, n_y])

                colpts_list.append([n_x, n_y])
                col_state = 1
                break

# Path Determination of Colour Ball
def colour_path(img, targ_x, targ_y):
    global colpts_list, prediction
    if len(colpts_list) > 0:
        dx = (targ_x - colpts_list[0][0])
        dy = (targ_y - colpts_list[0][1])
        for i in range(0, 1000):
            i /= 100
            n_x = int(targ_x + (dx * i))
            n_y = int(targ_y + (dy * i))
            if wall_touch_check(img, n_x, n_y):
                if inAnyPocket(n_x, n_y, 16):
                    prediction = 1
                    draw_points_1.append([colpts_list[0][0], colpts_list[0][1], n_x, n_y])

                    break
                else:
                    draw_points_1.append([colpts_list[0][0], colpts_list[0][1], n_x, n_y])
                    bounce_line(img, colpts_list[0], (n_x, n_y))
                    break




# Checking If the Colour Ball Touches The  Walls Of The Pool
def wall_touch_check(img, n_x, n_y):
    global targ_color
    if targ_color == 'b1':
        return n_x < 155 or n_x > 1120 or n_y < 100 or n_y > 540
    elif targ_color == 'ro':
        return n_x < 155 or n_x > 1130 or n_y < 100 or n_y > 540
    else:
        return n_x < 155 or n_x > 1120 or n_y < 95 or n_y > 520
    return False

# Predicting The Bounced Path of the Coloured Ball
def bounce_line(img, pt1, pt2):
    global prediction, line_ext_err
    bouncePt = pt2[0], pt2[1]
    if targ_color == 'y1':
        ref_pt = int((pt2[0] - pt1[0]) * 1.019) + pt2[0], pt1[1]
    elif targ_color == 'b1':
        ref_pt = int((pt2[0] - pt1[0]) * 1.4) + pt2[0], pt1[1]
    elif targ_color == 'ro':
        ref_pt = int((pt2[0] - pt1[0]) * 1.55) + pt2[0], pt1[1]
    else:
        ref_pt = int((pt2[0] - pt1[0]) * 0.7) + pt2[0], pt1[1]
    xE = (ref_pt[0] - bouncePt[0])
    yE = (ref_pt[1] - bouncePt[1])
    for i in range(0, 10000):
        i = i / 100
        n_x = int(ref_pt[0] + (xE * i))
        n_y = int(ref_pt[1] + (yE * i))
        if n_x > 155:
            line_ext_err.append([n_x, n_y])


        if wall_touch_check(img, n_x, n_y):
            if inAnyPocket(n_x, n_y, 16):
                prediction = 1
                draw_points_3.append([n_x, n_y, pt2[0], pt2[1]])
                break
            else:

                prediction = 0
                if line_ext_err:
                    line_dis= ((line_ext_err[-1][0] - n_x) ** 2 + (line_ext_err[-1][1] - n_y) ** 2) ** 0.5

                    if line_dis> 50:
                        draw_points_3.append([line_ext_err[-1][0], line_ext_err[-1][1], pt2[0], pt2[1]])
                    else:
                        draw_points_3.append([n_x, n_y, pt2[0], pt2[1]])
                break

# Checking If Colour Ball Falls In Any Of The Pockets
def pocket_touch_check(img, x1, y1, x2, y2):
    val1 = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    if val1 < 27:
        return True

# Checking the collision between Cue ball and Colour Ball
def ball_touch_check(img, n_x, n_y, targ_x, targ_y):
    global cueb_r, targ_r, targ_color, clearance
    val = ((n_x - targ_x) ** 2 + (n_y - targ_y) ** 2) ** 0.5
    if val - 15 <= cueb_r + targ_r <= val + 7 and targ_color == 'b1' and clearance == 5:
        return True
    if val - 15 <= cueb_r + targ_r + 0.99 <= val and targ_color == 'b1' and clearance != 5:
        return True
    if val - 5 <= cueb_r + targ_r - 3 <= val + 5 and targ_color == 'ro':
        return True
    if val - 10 <= cueb_r + targ_r <= val + 15 and targ_color == 'y1':
        return True
    if val - 11 <= cueb_r + targ_r <= val + 20 and targ_color == 'g1' and clearance == 2:
        return True
    if val - 11 <= cueb_r + targ_r - 18 <= val + 10 and targ_color == 'g1' and clearance == 6:
        return True

# Checking In Which Pocket The Ball Has Fallen
def inAnyPocket(x1, y1, radius):
    for i in range(6):
        if pocket_touch_check(img, x1, y1, poc_cor[i][0], poc_cor[i][1]):
            return True
            break

# Clearing The Stored Prediction
def clear_log():
    global cuel_list, cueb_list, targb_list, targb_pres, colpts_list, cuelb_pres, col_state, draw_points_1, draw_points_2, draw_points_3, line_ext_err
    col_state = 0
    colpts_list = []
    cuelb_pres = 0
    targb_pres = 0
    cueb_list = []
    cuel_list = []
    targb_list = []
    draw_points_1 = []
    draw_points_2 = []
    draw_points_3 = []
    line_ext_err = []


prediction = None
col_state = 0
colpts_list = []
cuelb_pres = 0
targb_pres = 0
cueb_list = []
cuel_list = []
targb_list = []
draw_points_1 = []
draw_points_2 = []
draw_points_3 = []
line_ext_err = []
targ_color = None
factor = 1
clearance = 0
skip_frame = 1
skip_frames = 0
frame_count = 0
colour_list = [(0,0,255),(0,255,0)]
color_find = ColorFinder(False)


while True:
    success, img = cap.read()
    if success:
        if skip_frame == 1:
            if frame_count == skip_frames:
                ret, frame = cap.read()
                frame_count = 0
            else:
                frame_count += 1
                continue
        img = cv2.resize(img, (1280, 720))
        imgColor_cue, mask_cue = color_find.update(img, hsv_cue)
        img, conf = findContours(img, imgpre=mask_cue, minArea=400, drawCon=0)
        if len(conf) >= 2:
            for con in conf:
                if con['radius'] < 18 and con['radius'] > 14 and con['bbox'][3] < 35 and con['bbox'][3] > 25:
                    cueb_list.append([con['center'][0], con['center'][1], con['radius']])
                    cueb_x, cueb_y = cueb_list[0][0], cueb_list[0][1]
                    cueb_area = con['area']
                    cueb_r = cueb_list[0][2]
                    cuelb_pres += 1
                elif con["radius"] > 24:
                    cuel_list.append((con['center'][0], con['center'][1]))
                    cuel_x, cuel_y = cuel_list[0]
                    cuel_area = con['area']
                    cuelb_pres += 1

                if cuelb_pres == 2:
                    break
        if len(cueb_list) > 3:
            x1, y1 = cueb_list[-1][0], cueb_list[-1][1]
            x2, y2 = cueb_list[-2][0], cueb_list[-2][1]
            x3, y3 = cueb_list[-3][0], cueb_list[-3][1]

            shot_cng1 = int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
            shot_cng2 = int(((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5)
            least_diff = abs(shot_cng2 - shot_cng1)

            if least_diff > 74:
                if factor != 4 and factor != 10 and factor != 11:
                    clearance += 1
                    clear_log()
                    prediction = None
                factor += 1

        if cuelb_pres == 2:
            for hsv_data in hsv_list:
                imgColor, mask = color_find.update(img, hsv_data)
                img, conf = findContours(img, mask, minArea=400, drawCon=0, color_check=1)
                if len(conf) > 0:
                    for con in conf:
                        if con['radius'] < 26 and con['radius'] > 10 and con['bbox'][3] < 35 and con['bbox'][3] > 20:
                            targb_list.append([con['center'][0], con['center'][1], con['radius'], hsv_data['color']])
                            targ_x, targ_y = targb_list[0][0], targb_list[0][1]
                            targ_r = targb_list[0][2]
                            targ_color = targb_list[0][3]
                            targb_pres = 1
                if targb_pres == 1:
                    break
        if targb_pres == 1 and cuelb_pres == 2:
            collision_check(img, cueb_x, cueb_y, cuel_x, cuel_y, targ_x, targ_y)
            colour_path(img, targ_x, targ_y)

        if len(draw_points_2) > 0:
            cv2.line(img, (draw_points_2[0][0], draw_points_2[0][1]), (draw_points_2[0][2], draw_points_2[0][3]),colour_list[prediction], 5)
            cv2.circle(img, (draw_points_2[0][0], draw_points_2[0][1]), 16, colour_list[prediction], cv2.FILLED)
            if len(draw_points_1) > 0:
                cv2.line(img, (draw_points_1[0][0], draw_points_1[0][1]), (draw_points_1[0][2], draw_points_1[0][3]), colour_list[prediction], 5)
                cv2.circle(img, (draw_points_1[0][0], draw_points_1[0][1]), 16, colour_list[prediction], cv2.FILLED)

            if len(draw_points_3) > 0:
                cv2.line(img, (draw_points_3[0][0], draw_points_3[0][1]), (draw_points_3[0][2], draw_points_3[0][3]),colour_list[prediction], 5)
                cv2.circle(img, (draw_points_1[0][0], draw_points_1[0][1]), 16, colour_list[prediction] , cv2.FILLED)
                cv2.circle(img, (draw_points_1[0][2], draw_points_1[0][3]), 16, colour_list[prediction], cv2.FILLED)

        if prediction is not None:
            if prediction == 1:
                cv2.rectangle(img, (cueb_x + 150, cueb_y - 30), (cueb_x + 300, cueb_y + 30), (0, 255, 0), -1)
                cv2.putText(img, "Prediction: IN", (cueb_x + 150, cueb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)
            elif prediction == 0:
                cv2.rectangle(img, (cueb_x + 150, cueb_y - 30), (cueb_x + 330, cueb_y + 30), (0, 0, 255), -1)
                cv2.putText(img, "Prediction: OUT", (cueb_x + 150, cueb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)


        cv2.imshow("Pool_Prediction", img)
        targb_pres = 0
        cuelb_pres = 0
        col_state = 0
        colpts_list = []
        capPred.write(img)
        if cv2.waitKey(1) & 0xFF == ord('o'):
            pass
    else:
        break