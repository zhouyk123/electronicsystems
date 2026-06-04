import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
import threading as thd

# ===== 关键参数 =====
# PID 参数
P, I, D = 0.05, 0, 1

# 速度参数
TURN_DUTY = 20          # 原地转弯速度（mode=='green'分支）
TURN_DUTY_FAST = 40     # 原地转弯速度（mode=='red'分支）
STRAIGHT_DUTY = 20      # 直行速度（mode=='green'分支）
STRAIGHT_DUTY_FAST = 40 # 直行速度（mode=='red'分支）
FORWARD_DUTY = 20       # forward()中的基础速度
RATIO = 1.025           # 左右轮补偿比
FINAL_RATIO = 1.04      # 最终冲线补偿比

# 面积阈值
LEAST_AREA_FIND_CUBE = 3000    # 找到魔方的最小面积
AREA_FOR_TURN = 80000          # 接近魔方停车的面积
LEAST_AREA_PASS_YELLOW = 1000  # 经过黄色的最小面积

# 时间参数（mode=='green' 分支）
FIND_TURN_TIME = 0.33      # 搜索时每次转动时间
TURN_LEFT_TIME = 0.6       # 绕行左转时间
STRAIGHT_TIME = 0.5        # 绕行直行时间
TURN_RIGHT_TIME = 0.7      # 绕行右转时间
PASS_STRAIGHT_TIME = 1.3   # 经过魔方直行时间
CIRCLE_TURN_TIME = 0.75    # 绕圈转弯时间
CIRCLE_STRAIGHT_TIME = 0.8 # 绕圈直行时间
FINAL_STRAIGHT_TIME = 5    # 最终冲线时间

# 时间参数（mode=='red' 分支）
RED_TURN_LEFT_TIME = 0.65      # 红色绕行左转时间（left位置）
RED_STRAIGHT_TIME = 0.5        # 红色绕行直行时间（left位置）
RED_TURN_RIGHT_TIME = 0.7      # 红色绕行右转时间（left位置）
RED_PASS_STRAIGHT_TIME = 1.3   # 红色经过魔方直行时间
RED_TURN_LEFT_TIME_DEFAULT = 0.6   # 红色绕行左转时间（非left位置）
RED_STRAIGHT_TIME_DEFAULT = 0.55   # 红色绕行直行时间（非left位置）
RED_TURN_RIGHT_TIME_DEFAULT = 0.72 # 红色绕行右转时间（非left位置）

# 搜索参数
MAX_TURN_COUNT = 80        # 最大搜索转动次数

# ===== 摄像头图像获取线程 =====

picSize = [Width, Height] = (640, 480)
Center = (Width / 2, Height / 2)
fps = 30
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
cap.set(cv2.CAP_PROP_FPS, fps)
frame = 0
fpsCount = 0
getPic = 0

def picShoot():
    global frame, cap, fpsCount, getPic
    try:
        while getPic == 1:
            ret, temp = cap.read()
            if ret:
                frame = temp
                fpsCount += 1
            else:
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
                cap.set(cv2.CAP_PROP_FPS, fps)
    except KeyboardInterrupt:
        cap.release()

picGet = thd.Thread(target=picShoot)

# ===== 图像处理 =====
# HSV空间下目标颜色, 最好根据卡纸颜色重设
low_red1 = np.array([0, 90,100])
high_red1 = np.array([7, 255, 255])
low_red2 = np.array([165, 90, 100])
high_red2 = np.array([180, 255, 255])
low_yellow = np.array([25, 80, 100])
high_yellow = np.array([35, 255, 255])
low_green = np.array([40,70, 70])
high_green = np.array([75, 255, 255])

def getImg_Mask(img, color):
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_Mask = 0
    if color == 'red':
        img_Mask_part1 = cv2.inRange(img_HSV, low_red1, high_red1)
        img_Mask_part2 = cv2.inRange(img_HSV, low_red2, high_red2)
        img_Mask = cv2.bitwise_or(img_Mask_part1, img_Mask_part2)
    elif color == 'yellow':
        img_Mask = cv2.inRange(img_HSV, low_yellow, high_yellow)
    elif color == 'green':
        img_Mask = cv2.inRange(img_HSV, low_green, high_green)
    return img_Mask

def get_Cube_center_area(img_Mask):
    _, edge, _ = cv2.findContours(img_Mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if edge:
        area = 0
        for subedge in edge:
            subarea = cv2.contourArea(subedge)
            if subarea >= 500:
                area += cv2.contourArea(subedge)

        if area < 1500:
            return 0,0,edge
        else:
            point_x = []
            for subedge in edge:
                for i in subedge:
                    x,_ = i[0]
                    point_x.append(x)
            center_x = np.mean(point_x)
            return center_x,area,edge
    else:
        return 0,0,edge

FREQUENCY = 80

# ===== 初始化 =====
def init():
    IO = [EA, I2, I1, EB, I4, I3] = [13, 19, 26, 16, 21, 17]
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
    GPIO.output([EA, I2, EB, I3], GPIO.LOW)
    GPIO.output([I1, I4], GPIO.HIGH)
    global pwma, pwmb
    pwma = GPIO.PWM(EA, FREQUENCY)
    pwmb = GPIO.PWM(EB, FREQUENCY)
    pwma.start(0)
    pwmb.start(0)

# ===== 控制行走部分 =====
class PID:
    (Kp, Ki, Kd) = (0, 0, 0)
    err_before = 0
    err_sum = 0

    def __init__(self, P, I, D):
        self.Kp, self.Ki, self.Kd = (P, I, D)

    def feedback(self, err):
        self.err_sum += err
        delta = self.Kp * err + self.Ki * self.err_sum + self.Kd * (err - self.err_before)
        self.err_before = err
        return delta

def go_straight(duty, ratio = RATIO):
    pwma.ChangeDutyCycle(duty)
    pwmb.ChangeDutyCycle(duty*ratio)

def stop():
    pwma.ChangeDutyCycle(0)
    pwmb.ChangeDutyCycle(0)

def turn(duty, delta):
    left = duty - delta
    right = duty + delta
    if left > 100:
        left = 100
    if left < 0:
        left = 0
    if right > 100:
        right = 100
    if right < 0:
        right = 0
    pwmb.ChangeDutyCycle(left)
    pwma.ChangeDutyCycle(right)

def turn_left(duty):
    left = 0
    right = duty
    pwmb.ChangeDutyCycle(left)
    pwma.ChangeDutyCycle(right)

def turn_right(duty):
    left = duty
    right = 0
    pwmb.ChangeDutyCycle(left)
    pwma.ChangeDutyCycle(right)

def forward(center_x):
    err = Center[0] - center_x
    delta = pidController.feedback(err)

    r = 4
    R = 8
    if delta > 0:
        delta = R if delta > R else delta
        delta = 0 if delta < r else delta
    else:
        delta = -R if delta < -R else delta
        delta = 0 if delta > -r else delta
    turn(FORWARD_DUTY, delta)

# ===== 顶层函数 =====
def detected_color(color):
    try:
        img = frame.astype(np.uint8)
        img_Mask = getImg_Mask(img, color)
        center_x, area, edge = get_Cube_center_area(img_Mask)
        if area < LEAST_AREA_FIND_CUBE:
            turnCount = 0
            flag = 0
            while turnCount < MAX_TURN_COUNT:
                turn_right(TURN_DUTY)
                time.sleep(FIND_TURN_TIME)
                stop()
                time.sleep(0.01)
                img = frame.astype(np.uint8)
                img_Mask = getImg_Mask(img, color)
                cv2.imshow("Current", frame)
                cv2.waitKey(1)
                center, area, edge = get_Cube_center_area(img_Mask)
                if (area > LEAST_AREA_FIND_CUBE):
                    flag = 1
                    break
                turnCount += 1
                if turnCount % 10 == 0:
                    print(f"搜索中... 已转 {turnCount} 次")
            stop()
            if (flag == 0):
                print("Fail to find cube")
                exit()
        print(f"Find {color}")
    except KeyboardInterrupt:
        exit()

def forward_color(color):
    while True:
        img = frame.astype(np.uint8)
        img_Mask = getImg_Mask(img, color)
        center_x, area, edge = get_Cube_center_area(img_Mask)
        cv2.imshow("Current", frame)
        cv2.waitKey(1)
        if area > 0 :
            forward(center_x)
        if area > AREA_FOR_TURN:
            stop()
            break

# ===== 测试部分 =====
def picTest1():
    img = cv2.imread("OIP-C.jpg")
    img_Mask = getImg_Mask(img, "green")
    center, area, edge = get_Cube_center_area(img_Mask)
    img_edge = cv2.drawContours(img, edge, -1, (0, 255, 0), 3)
    print(center, area)
    cv2.imshow("img", img_edge)
    cv2.waitKey(0)


if __name__ == '__main__':
    # 用于防止上次没清理端口
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(21,GPIO.OUT)
    GPIO.cleanup()
    # 初始化
    init()

    # 摄像线程
    getPic = 1
    picGet.start()

    # 控制朝向魔方移动的PID
    pidController = PID(P, I, D)

    time.sleep(5)

    mode = 'green' # 根据红色在前与在后设置两个模式
    """stop()之后务必加time.sleep(), 否则该stop()是无效的!!!"""
    if mode == 'red':
        red_position = "" # 红色魔方位置参数

        # 红色魔方
        detected_color('red')
        print("第1步：找红色 完成")

        forward_color('red')
        print("第2步：向红色前进 完成")
        stop()
        time.sleep(0.5)

        if red_position == "left":
            # Step1 左直右 -- 小车位于红色魔方左侧
            turn_left(TURN_DUTY_FAST)
            time.sleep(RED_TURN_LEFT_TIME)
            go_straight(TURN_DUTY_FAST)
            time.sleep(RED_STRAIGHT_TIME)
            stop()
            time.sleep(0.2)
            turn_right(TURN_DUTY_FAST)
            time.sleep(RED_TURN_RIGHT_TIME)
            stop()
            time.sleep(0.2)

            # Step2 直走 -- 小车走过红色魔方
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(RED_PASS_STRAIGHT_TIME)

            # Step3 右直左 -- 小车回到场地中央
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.7)
            stop()
            time.sleep(0.2)
            go_straight(TURN_DUTY_FAST)
            time.sleep(1.1)
            stop()
            time.sleep(0.2)
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.48)

            # Step4 直走一段距离方便下一步找黄色
            go_straight(FORWARD_DUTY)
            time.sleep(0.3)
            stop()
            time.sleep(0.2)
        else:
            # Step1 左直右 -- 小车位于红色魔方左侧
            turn_left(TURN_DUTY_FAST)
            time.sleep(RED_TURN_LEFT_TIME_DEFAULT)
            go_straight(TURN_DUTY_FAST)
            time.sleep(RED_STRAIGHT_TIME_DEFAULT)
            stop()
            time.sleep(0.2)
            turn_right(TURN_DUTY_FAST)
            time.sleep(RED_TURN_RIGHT_TIME_DEFAULT)
            stop()
            time.sleep(0.2)

            # Step2 直走 -- 小车走过红色魔方
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(RED_PASS_STRAIGHT_TIME)

            stop()
            time.sleep(0.2)

        print("第3步：从红色左边绕过 完成")

        # 黄色魔方
        stop()
        time.sleep(0.2)
        # 略左转, 识别左侧黄色魔方
        turn_left(TURN_DUTY_FAST)
        time.sleep(0.6)
        stop()
        time.sleep(0.2)

        detected_color('yellow')
        print("第4步：找黄色 完成")

        forward_color('yellow')
        print("第5步：向黄色前进 完成")
        stop()
        time.sleep(0.2)

        yellow_interval = "big"

        if yellow_interval == "little":
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.55)

            turn_left(TURN_DUTY_FAST)
            time.sleep(0.50)
        else:
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.7)
            stop()
            time.sleep(0.2)
            go_straight(TURN_DUTY_FAST)
            time.sleep(0.65)
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.48)

        # 直行经过黄色魔方
        go_straight(TURN_DUTY_FAST, 1.028)
        time.sleep(RED_PASS_STRAIGHT_TIME)
        stop()
        time.sleep(0.2)

        print("第6步：从黄色右边绕过 完成")

        # 绿色魔方
        green_position = "right"

        if green_position == "right":
            detected_color('green')
            print("第7步：找绿色 完成")

            forward_color('green')
            print("第8步：向绿色前进 完成")

            # Step1 右直左 -- 小车位于魔方右侧
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.72)
            stop()
            time.sleep(0.5)
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.6)

            turn_left(TURN_DUTY_FAST)
            time.sleep(0.6)
            stop()
            time.sleep(0.2)

            # Step2 直行 -- 通过魔方
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.85)
            stop()
            time.sleep(0.1)
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.1)
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.2)
        else:
            # 先判断视野中是否有绿色魔方
            img = frame.astype(np.uint8)
            img_Mask = getImg_Mask(img, "green")
            center_x, area, edge = get_Cube_center_area(img_Mask)
            if area < LEAST_AREA_FIND_CUBE:
                turn_left(TURN_DUTY_FAST)
                time.sleep(0.5)
            detected_color('green')
            print("第7步：找绿色 完成")

            forward_color('green')
            print("第8步：向绿色前进 完成")

            # Step1 右直左 -- 小车位于魔方右侧
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.72)
            stop()
            time.sleep(0.5)
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.60)

            turn_left(TURN_DUTY_FAST)
            time.sleep(0.42)
            stop()
            time.sleep(0.2)

            # Step2 直行 -- 通过魔方
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.85)
            stop()
            time.sleep(0.1)
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.1)
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.2)

        print("第9步：从绿色右边绕过 完成")

    else:
        # mode == 'green'
        detected_color('red')
        print("第1步：找红色 完成")

        forward_color('red')
        print("第2步：向红色前进 完成")
        stop()
        time.sleep(0.5)

        turn_left(TURN_DUTY)
        time.sleep(TURN_LEFT_TIME)
        go_straight(TURN_DUTY)
        time.sleep(STRAIGHT_TIME)
        turn_right(TURN_DUTY)
        time.sleep(TURN_RIGHT_TIME)
        stop()
        time.sleep(0.2)
        go_straight(TURN_DUTY, RATIO)
        time.sleep(PASS_STRAIGHT_TIME)
        stop()
        time.sleep(0.2)
        print("第3步：从红色左边绕过 完成")

        detected_color('green')
        print("第4步：找绿色 完成")

        forward_color('green')
        print("第5步：向绿色前进 完成")
        stop()
        time.sleep(0.5)

        for i in range(4):
            turn_left(TURN_DUTY)
            time.sleep(CIRCLE_TURN_TIME)
            stop()
            time.sleep(0.2)
            go_straight(TURN_DUTY)
            time.sleep(CIRCLE_STRAIGHT_TIME)
            stop()
            time.sleep(0.2)
        print("第6步：绕绿色转一圈 完成")

        stop()
        time.sleep(0.2)

        detected_color('yellow')
        print("第7步：找黄色 完成")

        forward_color('yellow')
        print("第8步：向黄色前进 完成")
        stop()
        time.sleep(0.5)

        turn_right(TURN_DUTY)
        time.sleep(TURN_RIGHT_TIME)
        go_straight(TURN_DUTY)
        time.sleep(STRAIGHT_TIME)
        turn_left(TURN_DUTY)
        time.sleep(0.55)
        stop()
        time.sleep(0.2)
        go_straight(TURN_DUTY, RATIO)
        time.sleep(PASS_STRAIGHT_TIME)
        stop()
        time.sleep(0.2)
        print("第9步：从黄色右边绕过 完成")

    go_straight(TURN_DUTY, FINAL_RATIO)
    time.sleep(FINAL_STRAIGHT_TIME)
    stop()
    print("第10步：冲线 完成")

    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
