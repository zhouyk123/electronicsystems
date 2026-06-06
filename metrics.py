import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import RPi.GPIO as GPIO
import threading as thd

# ===== 关键参数 =====
# 前进模式（朝前/朝后）
MODE_ = 1  # 0, 1
MODE_L = 0  # 0, 1
MODE_R = 0

# 速度参数
TURN_DUTY = 20          # 原地转弯速度（mode=='green'分支）
TURN_DUTY_FAST = 40     # 原地转弯速度（mode=='red'分支）
STRAIGHT_DUTY = 20      # 直行速度（mode=='green'分支）
STRAIGHT_DUTY_FAST = 40 # 直行速度（mode=='red'分支）
FORWARD_DUTY = 20       # forward()中的基础 duty
RATIO = 1.025           # 左右轮补偿比
FINAL_RATIO = 1.04      # 最终冲线补偿比

# 转向速度闭环参数（参考 pid_control.py）
SPEED_SAMPLE_TIME = 0.1
SPEED_PULSE_PER_REV = 585.0
TURN_SPEED_TARGET = 1.9
SPEED_PID = {"P":30, "I":0.06, "D":20}
FORWARD_PID = {"P":0.1, "I":0, "D":1}
ANGLE_FACTOR = 90

# 面积阈值
LEAST_AREA_FIND_CUBE = 3000    # 找到魔方的最小面积
AREA_FOR_TURN = 80000          # 接近魔方停车的面积
LEAST_AREA_PASS_YELLOW = 1000  # 经过黄色的最小面积

# 时间参数（mode=='green' 分支）
FIND_TURN_TIME = 0.33      # 搜索时每次转动时间
FIND_TURN_REST_TIME = 1
INTERVAL_SLEEP_TIME = 1    # 操作间休息时间


# 搜索参数
MAX_TURN_COUNT = 80        # 最大搜索转动次数
WAIT_EVERY_TURN = 0        # 0 for waiting for keyboard input and n >= 0 for waiting for n ms

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
getSpeed = 0
lspeed = 0
rspeed = 0
speed_lcounter = 0  # 用于测速的 counter
speed_rcounter = 0
move_lcounter = 0  # 用于计数的 counter
move_rcounter = 0
speedGet = None
speedLock = thd.Lock()
turnPidLock = thd.Lock()
turnPidMode = None
turnPidController = None

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

def speed_callback(channel):
    global speed_lcounter, speed_rcounter, move_lcounter, move_rcounter, threshold, triggered
    with speedLock:
        if channel == LS:
            speed_lcounter += 1
            move_lcounter += 1
            if threshold > 0 and move_lcounter >= threshold * SPEED_PULSE_PER_REV:
                triggered = True
        elif channel == RS:
            speed_rcounter += 1
            move_rcounter += 1
            if threshold > 0 and move_rcounter >= threshold * SPEED_PULSE_PER_REV:
                triggered = True


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
    # 查找二值 mask 中的外轮廓
    result = cv2.findContours(
        img_Mask,
        cv2.RETR_EXTERNAL,          # 只检测最外层轮廓
        cv2.CHAIN_APPROX_SIMPLE     # 压缩轮廓点，只保留必要点
    )
    # 兼容 OpenCV 3 和 OpenCV 4 的不同返回格式
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result

    # 没有找到轮廓，直接返回 0
    if not contours:
        return 0, 0, contours

    area = 0
    m00 = 0
    m10 = 0

    # 遍历所有轮廓，保留面积足够大的轮廓
    for contour in contours:
        subarea = cv2.contourArea(contour)
        # 过滤小轮廓，避免噪声影响中心计算
        if subarea >= 500:
            area += subarea
            # 计算当前轮廓的图像矩
            M = cv2.moments(contour)
            # M["m00"] 可以理解为该轮廓的面积
            # M["m10"] 用于计算 x 方向质心
            m00 += M["m00"]
            m10 += M["m10"]
    # 有效轮廓总面积太小，认为没有检测到目标
    if area < 1500 or m00 == 0:
        return 0, 0, contours
    # 整体质心的 x 坐标
    center_x = m10 / m00
    # 返回目标水平中心、有效轮廓总面积、原始轮廓列表
    return center_x, area, contours


def detected_color(color):
    try:
        img = frame.astype(np.uint8)
        img_Mask = getImg_Mask(img, color)
        center_x, area, edge = get_Cube_center_area(img_Mask)
        if area < LEAST_AREA_FIND_CUBE:
            turnCount = 0
            flag = 0
            while turnCount < MAX_TURN_COUNT:
                turn_right(TURN_DUTY, 45)
                time.sleep(FIND_TURN_TIME)
                stop()
                time.sleep(FIND_TURN_REST_TIME)
                img = frame.astype(np.uint8)
                img_Mask = getImg_Mask(img, color)
                cv2.imshow("Current", frame)
                cv2.waitKey(WAIT_EVERY_TURN)
                center, area, edge = get_Cube_center_area(img_Mask)
                if (area > LEAST_AREA_FIND_CUBE):
                    flag = 1
                    break
                turnCount += 1
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


FREQUENCY = 80

# ===== 初始化 =====
def speedShoot():
    """
    原地转弯的闭环控制，包括测速和调速
    """
    global lspeed, rspeed, speed_lcounter, speed_rcounter, getSpeed
    while getSpeed == 1:
        time.sleep(SPEED_SAMPLE_TIME)
        with speedLock:
            left_count = speed_lcounter
            right_count = speed_rcounter
            speed_lcounter = 0
            speed_rcounter = 0

        lspeed = left_count / SPEED_PULSE_PER_REV
        rspeed = right_count / SPEED_PULSE_PER_REV

        with turnPidLock:
            mode = turnPidMode

        if mode in ("left", "right"):  # 只在左转或右转时才依靠这里 PID 调节
            right_duty = turnPidController_right.update(rspeed)
            left_duty = turnPidController_left.update(lspeed)
            pwma.ChangeDutyCycle(left_duty)
            pwmb.ChangeDutyCycle(right_duty)


def set_motor_mode(pin:list[int], direction:int):
    assert direction in (0,1,2)
    if direction == 0:
        GPIO.output(pin[1], GPIO.HIGH)
        GPIO.output(pin[2], GPIO.LOW)
    elif direction == 1:
        GPIO.output(pin[1], GPIO.LOW)
        GPIO.output(pin[2], GPIO.HIGH)


def init_counter():
    global move_lcounter, move_rcounter, threshold, triggered
    with speedLock:
        move_lcounter = 0
        move_rcounter = 0
        threshold = 0
        triggered = False


def init():
    print("[初始化] GPIO清理...")
    GPIO.cleanup()
    IO = [EA, I2, I1, B2A, EB, I4, I3, B1A] = [13, 19, 26, 6, 16, 21, 17, 12]
    GPIO.setmode(GPIO.BCM)

    GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
    GPIO.setup([B2A, B1A], GPIO.IN)

    GPIO.output([EA, EB], GPIO.LOW)

    global left_pin, right_pin, LS, RS
    if MODE_ == 0:
        left_pin = [EA, I1, I2, B2A]
        right_pin= [EB, I4, I3, B1A]
        LS, RS = B2A, B1A
    else:
        left_pin = [EB, I3, I4, B1A]
        right_pin = [EA, I2, I1, B2A]
        LS, RS = B1A, B2A
    set_motor_mode(left_pin, MODE_L)
    set_motor_mode(right_pin, MODE_R)
    

    global pwma, pwmb, getSpeed, speedGet, threshold, triggered
    threshold = 0
    triggered = False  # 马达计数达到多少时触发外部函数
    pwma = GPIO.PWM(left_pin[0], FREQUENCY)
    pwmb = GPIO.PWM(right_pin[0], FREQUENCY)
    pwma.start(0)
    pwmb.start(0)
    GPIO.add_event_detect(LS, GPIO.RISING, callback=speed_callback)
    GPIO.add_event_detect(RS, GPIO.RISING, callback=speed_callback)
    getSpeed = 1
    speedGet = thd.Thread(target=speedShoot, daemon=True)
    speedGet.start()

# ===== 控制行走部分 =====
class WheelSpeedPID:
    def __init__(self, P, I, D, target, lb=0, ub=100):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.target = target
        self.err_before = 0
        self.err_sum = 0
        self.u = 0
        self.lb = lb  # lower bound
        self.ub = ub  # upper bound
        assert self.lb < self.ub

    def update(self, feedback):
        err = self.target - feedback
        self.err_sum += err
        self.u = self.Kp * err + self.Ki * self.err_sum + self.Kd * (err - self.err_before)
        self.err_before = err
        if self.ub is not None and self.u > self.ub:
            self.u = self.ub
        elif self.lb is not None and self.u < self.lb:
            self.u = self.lb
        return self.u


def set_turn_pid_mode(mode, duty=0):
    global turnPidMode, turnPidController, turnPidController_left, turnPidController_right
    with turnPidLock:
        if mode is None:
            turnPidMode = None
            turnPidController = None
            turnPidController_left = None
            turnPidController_right = None
            return

        target_speed = TURN_SPEED_TARGET
        if mode in ('left', 'right'):
            turnPidController = None
            turnPidController_left = WheelSpeedPID(**SPEED_PID, target=target_speed)
            turnPidController_right = WheelSpeedPID(**SPEED_PID, target=target_speed)
        turnPidMode = mode


def go_straight(duty, ratio=RATIO, dist=1):
    init_counter()
    global threshold, triggered
    threshold = dist

    set_motor_mode(left_pin, MODE_L)
    set_motor_mode(right_pin, MODE_R)
    set_turn_pid_mode(None)

    pwma.ChangeDutyCycle(duty)
    pwmb.ChangeDutyCycle(duty*ratio)

    while True:
        if triggered:
            brake()
            return

def stop(stop_time=0.5):
    set_turn_pid_mode(None)
    pwma.ChangeDutyCycle(0)
    pwmb.ChangeDutyCycle(0)
    time.sleep(stop_time)

def brake(brake_time=0.5):
    set_turn_pid_mode(None)
    pwma.ChangeDutyCycle(0)
    pwmb.ChangeDutyCycle(0)
    GPIO.output(left_pin[:3], GPIO.LOW)
    GPIO.output(right_pin[:3], GPIO.LOW)
    time.sleep(brake_time)

def turn_left(duty=TURN_DUTY, angle=90):
    init_counter()
    global threshold, triggered
    threshold = angle / ANGLE_FACTOR

    set_motor_mode(left_pin, not MODE_L)
    set_motor_mode(right_pin, MODE_R)
    set_turn_pid_mode('left', duty)
    left = duty
    right = duty
    pwma.ChangeDutyCycle(left)
    pwmb.ChangeDutyCycle(right)

    while True:
        if triggered:
            brake()
            return

def turn_right(duty=TURN_DUTY, angle=90):
    init_counter()
    global threshold, triggered
    threshold = angle / ANGLE_FACTOR

    set_motor_mode(left_pin, MODE_L)
    set_motor_mode(right_pin, not MODE_R)
    set_turn_pid_mode('right', duty)
    left = duty
    right = duty
    pwma.ChangeDutyCycle(left)
    pwmb.ChangeDutyCycle(right)

    while True:
        if triggered:
            brake()
            return

def forward(center_x):
    def turn(duty, delta):
        set_turn_pid_mode(None)
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
        pwma.ChangeDutyCycle(left)
        pwmb.ChangeDutyCycle(right)
    err = Center[0] - center_x
    delta = pidController.update(err)
    turn(FORWARD_DUTY, delta)

"""
所有可用的动作：
go_straight(duty, ratio, dist)
turn_left(duty, angel)
turn_right(duty, angle)
detect_color(color)
forward_color(color)
stop(stop_time)
brake(brake_time)
"""


if __name__ == '__main__':
    init()
    print("[初始化] 电机PWM就绪")

    # 启动摄像头采集线程（持续更新全局变量frame）
    getPic = 1
    picGet.start()
    print("[初始化] 摄像头线程已启动")

    # 初始化PID控制器，用于前进时的方向修正
    pidController = WheelSpeedPID(**FORWARD_PID, target=0, lb=-8, ub=8)
    print(f"[初始化] PID控制器就绪 ({FORWARD_PID})")

    # 等待5秒，确保摄像头稳定出图
    print("[初始化] 等待摄像头稳定(5秒)...")
    time.sleep(5)
    print("[初始化] 初始化完成，开始执行任务")
