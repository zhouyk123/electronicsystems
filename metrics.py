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

# 转向速度闭环参数（参考 pid_control.py）
LS, RS = 6, 12
SPEED_SAMPLE_TIME = 0.1
SPEED_PULSE_PER_REV = 585.0
TURN_SPEED_PER_DUTY = 1.9 / TURN_DUTY
LEFT_TURN_PID = (30, 0.06, 20)
RIGHT_TURN_PID = (40, 0.01, 23)

# 面积阈值
LEAST_AREA_FIND_CUBE = 3000    # 找到魔方的最小面积
AREA_FOR_TURN = 80000          # 接近魔方停车的面积
LEAST_AREA_PASS_YELLOW = 1000  # 经过黄色的最小面积

# 时间参数（mode=='green' 分支）
FIND_TURN_TIME = 0.33      # 搜索时每次转动时间
FIND_TURN_REST_TIME = 1
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
getSpeed = 0
lspeed = 0
rspeed = 0
lcounter = 0
rcounter = 0
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
    global lcounter, rcounter
    with speedLock:
        if channel == LS:
            lcounter += 1
        elif channel == RS:
            rcounter += 1

def speedShoot():
    global lspeed, rspeed, lcounter, rcounter, getSpeed
    while getSpeed == 1:
        time.sleep(SPEED_SAMPLE_TIME)
        with speedLock:
            left_count = lcounter
            right_count = rcounter
            lcounter = 0
            rcounter = 0

        lspeed = left_count / SPEED_PULSE_PER_REV
        rspeed = right_count / SPEED_PULSE_PER_REV

        with turnPidLock:
            mode = turnPidMode
            controller = turnPidController

        if mode == 'left':
            right_duty = controller.update(rspeed)
            pwmb.ChangeDutyCycle(0)
            pwma.ChangeDutyCycle(right_duty)
        elif mode == 'right':
            left_duty = controller.update(lspeed)
            pwmb.ChangeDutyCycle(left_duty)
            pwma.ChangeDutyCycle(0)

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
    GPIO.setup([LS, RS], GPIO.IN)
    GPIO.output([EA, I2, EB, I3], GPIO.LOW)
    GPIO.output([I1, I4], GPIO.HIGH)
    global pwma, pwmb, getSpeed, speedGet
    pwma = GPIO.PWM(EA, FREQUENCY)
    pwmb = GPIO.PWM(EB, FREQUENCY)
    pwma.start(0)
    pwmb.start(0)
    GPIO.add_event_detect(LS, GPIO.RISING, callback=speed_callback)
    GPIO.add_event_detect(RS, GPIO.RISING, callback=speed_callback)
    getSpeed = 1
    speedGet = thd.Thread(target=speedShoot, daemon=True)
    speedGet.start()

# ===== 控制行走部分 =====
class WheelSpeedPID:
    def __init__(self, P, I, D, target_speed):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.target_speed = target_speed
        self.err_before = 0
        self.err_sum = 0
        self.u = 0

    def update(self, feedback_speed):
        err = self.target_speed - feedback_speed
        self.err_sum += err
        self.u = self.Kp * err + self.Ki * self.err_sum + self.Kd * (err - self.err_before)
        self.err_before = err
        if self.u > 100:
            self.u = 100
        elif self.u < 0:
            self.u = 0
        return self.u

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

def set_turn_pid_mode(mode, duty=0):
    global turnPidMode, turnPidController
    with turnPidLock:
        if mode is None:
            turnPidMode = None
            turnPidController = None
            return

        target_speed = duty * TURN_SPEED_PER_DUTY
        if mode == 'left':
            turnPidController = WheelSpeedPID(*RIGHT_TURN_PID, target_speed)
        elif mode == 'right':
            turnPidController = WheelSpeedPID(*LEFT_TURN_PID, target_speed)
        turnPidMode = mode

def go_straight(duty, ratio = RATIO):
    set_turn_pid_mode(None)
    pwma.ChangeDutyCycle(duty)
    pwmb.ChangeDutyCycle(duty*ratio)

def stop():
    set_turn_pid_mode(None)
    pwma.ChangeDutyCycle(0)
    pwmb.ChangeDutyCycle(0)

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
    pwmb.ChangeDutyCycle(left)
    pwma.ChangeDutyCycle(right)

def turn_left(duty):
    set_turn_pid_mode('left', duty)
    left = 0
    right = duty
    pwmb.ChangeDutyCycle(left)
    pwma.ChangeDutyCycle(right)

def turn_right(duty):
    set_turn_pid_mode('right', duty)
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
                time.sleep(FIND_TURN_REST_TIME)
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
    # ========== 初始化阶段 ==========
    # 清理上次残留的GPIO状态，防止端口冲突
    print("[初始化] GPIO清理...")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(21,GPIO.OUT)
    GPIO.cleanup()
    # 初始化电机PWM
    init()
    print("[初始化] 电机PWM就绪")

    # 启动摄像头采集线程（持续更新全局变量frame）
    getPic = 1
    picGet.start()
    print("[初始化] 摄像头线程已启动")

    # 初始化PID控制器，用于前进时的方向修正
    pidController = PID(P, I, D)
    print(f"[初始化] PID控制器就绪 (P={P}, I={I}, D={D})")

    # 等待5秒，确保摄像头稳定出图
    print("[初始化] 等待摄像头稳定(5秒)...")
    time.sleep(5)
    print("[初始化] 初始化完成，开始执行任务")

    # ========== 模式选择 ==========
    # mode='red': 红色魔方在前，绕行顺序为 红->黄->绿
    # mode='green': 绿色模式，绕行顺序为 红->绿(绕圈)->黄
    mode = 'green'
    print(f"[配置] 当前模式: {mode}")
    """stop()之后务必加time.sleep(), 否则该stop()是无效的!!!"""

    if mode == 'red':
        # ==================== RED模式 ====================
        # 红色魔方位置参数: "left"表示魔方偏左, 其他值表示默认位置
        red_position = ""
        print(f"[配置] red_position={red_position}")

        # ------ 第1步: 搜索红色魔方 (L285) ------
        # 原地右转扫描，直到摄像头识别到红色区域面积 > LEAST_AREA_FIND_CUBE
        print("[RED-第1步] 开始搜索红色魔方...")
        detected_color('red')
        print("[RED-第1步] 找红色 完成")

        # ------ 第2步: PID前进逼近红色魔方 (L289) ------
        # 持续前进直到红色面积 > AREA_FOR_TURN，表示已靠近魔方
        print("[RED-第2步] PID前进逼近红色魔方...")
        forward_color('red')
        print("[RED-第2步] 向红色前进 完成，停车")
        stop()
        time.sleep(0.5)

        # ------ 第3步: 从红色魔方左侧绕过 (L293-L345) ------
        if red_position == "left":
            # === red_position=="left" 的绕行路径 ===
            # Step3-1: 左转避开魔方 (L295)
            print(f"[RED-第3步] left路径: 左转 duty={TURN_DUTY_FAST} time={RED_TURN_LEFT_TIME}s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(RED_TURN_LEFT_TIME)
            # Step3-1: 直行绕到魔方侧面 (L297)
            print(f"[RED-第3步] left路径: 直行 duty={TURN_DUTY_FAST} time={RED_STRAIGHT_TIME}s")
            go_straight(TURN_DUTY_FAST)
            time.sleep(RED_STRAIGHT_TIME)
            stop()
            time.sleep(0.2)
            # Step3-1: 右转回正车头 (L301)
            print(f"[RED-第3步] left路径: 右转回正 duty={TURN_DUTY_FAST} time={RED_TURN_RIGHT_TIME}s")
            turn_right(TURN_DUTY_FAST)
            time.sleep(RED_TURN_RIGHT_TIME)
            stop()
            time.sleep(0.2)

            # Step3-2: 直行通过红色魔方 (L307)
            print(f"[RED-第3步] left路径: 直行通过魔方 duty={TURN_DUTY_FAST} ratio={RATIO} time={RED_PASS_STRAIGHT_TIME}s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(RED_PASS_STRAIGHT_TIME)

            # Step3-3: 右转+直行+左转，回到场地中央 (L311-L320)
            print(f"[RED-第3步] left路径: 右转回中央 time=0.7s")
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.7)
            stop()
            time.sleep(0.2)
            print(f"[RED-第3步] left路径: 直行回中央 time=1.1s")
            go_straight(TURN_DUTY_FAST)
            time.sleep(1.1)
            stop()
            time.sleep(0.2)
            print(f"[RED-第3步] left路径: 左转回正 time=0.48s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.48)

            # Step3-4: 短距离直行，为下一步找黄色做准备 (L323)
            print(f"[RED-第3步] left路径: 短距直行准备找黄 duty={FORWARD_DUTY} time=0.3s")
            go_straight(FORWARD_DUTY)
            time.sleep(0.3)
            stop()
            time.sleep(0.2)
        else:
            # === red_position为默认值的绕行路径 ===
            # Step3-1: 左转避开魔方 (L330)
            print(f"[RED-第3步] 默认路径: 左转 duty={TURN_DUTY_FAST} time={RED_TURN_LEFT_TIME_DEFAULT}s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(RED_TURN_LEFT_TIME_DEFAULT)
            # Step3-1: 直行绕到魔方侧面 (L332)
            print(f"[RED-第3步] 默认路径: 直行 duty={TURN_DUTY_FAST} time={RED_STRAIGHT_TIME_DEFAULT}s")
            go_straight(TURN_DUTY_FAST)
            time.sleep(RED_STRAIGHT_TIME_DEFAULT)
            stop()
            time.sleep(0.2)
            # Step3-1: 右转回正车头 (L336)
            print(f"[RED-第3步] 默认路径: 右转回正 duty={TURN_DUTY_FAST} time={RED_TURN_RIGHT_TIME_DEFAULT}s")
            turn_right(TURN_DUTY_FAST)
            time.sleep(RED_TURN_RIGHT_TIME_DEFAULT)
            stop()
            time.sleep(0.2)

            # Step3-2: 直行通过红色魔方 (L341)
            print(f"[RED-第3步] 默认路径: 直行通过魔方 duty={TURN_DUTY_FAST} ratio={RATIO} time={RED_PASS_STRAIGHT_TIME}s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(RED_PASS_STRAIGHT_TIME)

            stop()
            time.sleep(0.2)

        print("[RED-第3步] 从红色左边绕过 完成")

        # ------ 第4步: 搜索黄色魔方 (L350-L356) ------
        # 先略微左转，让摄像头朝向左侧黄色魔方方向
        print("[RED-第4步] 左转调整朝向，准备找黄色...")
        stop()
        time.sleep(0.2)
        turn_left(TURN_DUTY_FAST)
        time.sleep(0.6)
        stop()
        time.sleep(0.2)

        # 原地扫描找黄色
        print("[RED-第4步] 开始搜索黄色魔方...")
        detected_color('yellow')
        print("[RED-第4步] 找黄色 完成")

        # ------ 第5步: PID前进逼近黄色魔方 (L361) ------
        print("[RED-第5步] PID前进逼近黄色魔方...")
        forward_color('yellow')
        print("[RED-第5步] 向黄色前进 完成，停车")
        stop()
        time.sleep(0.2)

        # ------ 第6步: 从黄色魔方右侧绕过 (L366-L388) ------
        # yellow_interval: "little"表示黄色离得近用小幅绕行, "big"表示大幅绕行
        yellow_interval = "big"
        print(f"[RED-第6步] yellow_interval={yellow_interval}，开始绕黄色魔方")

        if yellow_interval == "little":
            # 小幅绕行: 右转+左转即可通过 (L369-L373)
            print("[RED-第6步] 小幅绕行: 右转 time=0.55s")
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.55)
            print("[RED-第6步] 小幅绕行: 左转 time=0.50s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.50)
        else:
            # 大幅绕行: 右转+直行+左转 (L375-L382)
            print("[RED-第6步] 大幅绕行: 右转 time=0.7s")
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.7)
            stop()
            time.sleep(0.2)
            print("[RED-第6步] 大幅绕行: 直行 time=0.65s")
            go_straight(TURN_DUTY_FAST)
            time.sleep(0.65)
            print("[RED-第6步] 大幅绕行: 左转 time=0.48s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.48)

        # 直行经过黄色魔方（使用特殊补偿比1.028）(L385)
        print(f"[RED-第6步] 直行通过黄色魔方 duty={TURN_DUTY_FAST} ratio=1.028 time={RED_PASS_STRAIGHT_TIME}s")
        go_straight(TURN_DUTY_FAST, 1.028)
        time.sleep(RED_PASS_STRAIGHT_TIME)
        stop()
        time.sleep(0.2)

        print("[RED-第6步] 从黄色右边绕过 完成")

        # ------ 第7-9步: 绿色魔方绕行 (L393-L461) ------
        # green_position: "right"表示绿色魔方在右侧
        green_position = "right"
        print(f"[RED-第7步] green_position={green_position}")

        if green_position == "right":
            # === 绿色魔方在右侧的路径 ===
            # 第7步: 搜索绿色魔方 (L396)
            print("[RED-第7步] 开始搜索绿色魔方(右侧路径)...")
            detected_color('green')
            print("[RED-第7步] 找绿色 完成")

            # 第8步: PID前进逼近绿色魔方 (L399)
            print("[RED-第8步] PID前进逼近绿色魔方...")
            forward_color('green')
            print("[RED-第8步] 向绿色前进 完成")

            # 第9步-Step1: 右转+直行+左转，绕到魔方右侧 (L403-L412)
            print("[RED-第9步] Step1: 右转 time=0.72s")
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.72)
            stop()
            time.sleep(0.5)
            print(f"[RED-第9步] Step1: 直行 duty={TURN_DUTY_FAST} ratio={RATIO} time=0.6s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.6)
            print("[RED-第9步] Step1: 左转 time=0.6s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.6)
            stop()
            time.sleep(0.2)

            # 第9步-Step2: 直行通过魔方+微调方向 (L416-L422)
            print(f"[RED-第9步] Step2: 直行通过魔方 duty={TURN_DUTY_FAST} ratio={RATIO} time=0.85s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.85)
            stop()
            time.sleep(0.1)
            print("[RED-第9步] Step2: 微调左转 time=0.1s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.1)
            print("[RED-第9步] Step2: 继续直行 time=0.2s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.2)
        else:
            # === 绿色魔方不在右侧的路径 ===
            # 先检查视野中是否已有绿色，没有则左转搜索 (L426-L431)
            img = frame.astype(np.uint8)
            img_Mask = getImg_Mask(img, "green")
            center_x, area, edge = get_Cube_center_area(img_Mask)
            print(f"[RED-第7步] 预检测绿色面积={area}，阈值={LEAST_AREA_FIND_CUBE}")
            if area < LEAST_AREA_FIND_CUBE:
                print("[RED-第7步] 未发现绿色，左转搜索 time=0.5s")
                turn_left(TURN_DUTY_FAST)
                time.sleep(0.5)

            # 第7步: 搜索绿色魔方 (L432)
            print("[RED-第7步] 开始搜索绿色魔方(非右侧路径)...")
            detected_color('green')
            print("[RED-第7步] 找绿色 完成")

            # 第8步: PID前进逼近绿色魔方 (L435)
            print("[RED-第8步] PID前进逼近绿色魔方...")
            forward_color('green')
            print("[RED-第8步] 向绿色前进 完成")

            # 第9步-Step1: 右转+直行+左转，绕到魔方右侧 (L439-L448)
            print("[RED-第9步] Step1: 右转 time=0.72s")
            turn_right(TURN_DUTY_FAST)
            time.sleep(0.72)
            stop()
            time.sleep(0.5)
            print(f"[RED-第9步] Step1: 直行 duty={TURN_DUTY_FAST} ratio={RATIO} time=0.60s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.60)
            print("[RED-第9步] Step1: 左转 time=0.42s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.42)
            stop()
            time.sleep(0.2)

            # 第9步-Step2: 直行通过魔方+微调方向 (L452-L458)
            print(f"[RED-第9步] Step2: 直行通过魔方 duty={TURN_DUTY_FAST} ratio={RATIO} time=0.85s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.85)
            stop()
            time.sleep(0.1)
            print("[RED-第9步] Step2: 微调左转 time=0.1s")
            turn_left(TURN_DUTY_FAST)
            time.sleep(0.1)
            print("[RED-第9步] Step2: 继续直行 time=0.2s")
            go_straight(TURN_DUTY_FAST, RATIO)
            time.sleep(0.2)

        print("[RED-第9步] 从绿色右边绕过 完成")

    else:
        # ==================== GREEN模式 ====================
        # 绕行顺序: 红色(左绕) -> 绿色(绕圈) -> 黄色(右绕) -> 冲线

        # ------ 第1步: 搜索红色魔方 (L465) ------
        # 原地右转扫描，直到识别到红色区域面积 > LEAST_AREA_FIND_CUBE
        print("[GREEN-第1步] 开始搜索红色魔方...")
        detected_color('red')
        print("[GREEN-第1步] 找红色 完成")

        # ------ 第2步: PID前进逼近红色魔方 (L468) ------
        # 持续前进直到红色面积 > AREA_FOR_TURN
        print("[GREEN-第2步] PID前进逼近红色魔方...")
        forward_color('red')
        print("[GREEN-第2步] 向红色前进 完成，停车")
        stop()
        time.sleep(0.5)

        # ------ 第3步: 从红色魔方左侧绕过 (L473-L484) ------
        # 左转避开魔方 (L473)
        print(f"[GREEN-第3步] 左转避开 duty={TURN_DUTY} time={TURN_LEFT_TIME}s")
        turn_left(TURN_DUTY)
        time.sleep(TURN_LEFT_TIME)
        # 直行绕到魔方侧面 (L475)
        print(f"[GREEN-第3步] 直行绕侧面 duty={TURN_DUTY} time={STRAIGHT_TIME}s")
        go_straight(TURN_DUTY)
        time.sleep(STRAIGHT_TIME)
        # 右转回正车头 (L477)
        print(f"[GREEN-第3步] 右转回正 duty={TURN_DUTY} time={TURN_RIGHT_TIME}s")
        turn_right(TURN_DUTY)
        time.sleep(TURN_RIGHT_TIME)
        stop()
        time.sleep(0.2)
        # 直行通过红色魔方 (L481)
        print(f"[GREEN-第3步] 直行通过魔方 duty={TURN_DUTY} ratio={RATIO} time={PASS_STRAIGHT_TIME}s")
        go_straight(TURN_DUTY, RATIO)
        time.sleep(PASS_STRAIGHT_TIME)
        stop()
        time.sleep(0.2)
        print("[GREEN-第3步] 从红色左边绕过 完成")

        # ------ 第4步: 搜索绿色魔方 (L487) ------
        print("[GREEN-第4步] 开始搜索绿色魔方...")
        detected_color('green')
        print("[GREEN-第4步] 找绿色 完成")

        # ------ 第5步: PID前进逼近绿色魔方 (L490) ------
        print("[GREEN-第5步] PID前进逼近绿色魔方...")
        forward_color('green')
        print("[GREEN-第5步] 向绿色前进 完成，停车")
        stop()
        time.sleep(0.5)

        # ------ 第6步: 绕绿色魔方转一圈 (L495-L503) ------
        # 四次"左转+直行"组成一个方形绕圈路径
        print(f"[GREEN-第6步] 开始绕圈(4次), 转弯time={CIRCLE_TURN_TIME}s, 直行time={CIRCLE_STRAIGHT_TIME}s")
        for i in range(4):
            # 左转90度 (L496)
            print(f"[GREEN-第6步] 绕圈第{i+1}/4次: 左转...")
            turn_left(TURN_DUTY)
            time.sleep(CIRCLE_TURN_TIME)
            stop()
            time.sleep(0.2)
            # 直行一段（方形的一条边）(L500)
            print(f"[GREEN-第6步] 绕圈第{i+1}/4次: 直行...")
            go_straight(TURN_DUTY)
            time.sleep(CIRCLE_STRAIGHT_TIME)
            stop()
            time.sleep(0.2)
        print("[GREEN-第6步] 绕绿色转一圈 完成")

        stop()
        time.sleep(0.2)

        # ------ 第7步: 搜索黄色魔方 (L509) ------
        print("[GREEN-第7步] 开始搜索黄色魔方...")
        detected_color('yellow')
        print("[GREEN-第7步] 找黄色 完成")

        # ------ 第8步: PID前进逼近黄色魔方 (L512) ------
        print("[GREEN-第8步] PID前进逼近黄色魔方...")
        forward_color('yellow')
        print("[GREEN-第8步] 向黄色前进 完成，停车")
        stop()
        time.sleep(0.5)

        # ------ 第9步: 从黄色魔方右侧绕过 (L517-L528) ------
        # 右转避开魔方 (L517)
        print(f"[GREEN-第9步] 右转避开 duty={TURN_DUTY} time={TURN_RIGHT_TIME}s")
        turn_right(TURN_DUTY)
        time.sleep(TURN_RIGHT_TIME)
        # 直行绕到魔方侧面 (L519)
        print(f"[GREEN-第9步] 直行绕侧面 duty={TURN_DUTY} time={STRAIGHT_TIME}s")
        go_straight(TURN_DUTY)
        time.sleep(STRAIGHT_TIME)
        # 左转回正车头 (L521)
        print(f"[GREEN-第9步] 左转回正 duty={TURN_DUTY} time=0.55s")
        turn_left(TURN_DUTY)
        time.sleep(0.55)
        stop()
        time.sleep(0.2)
        # 直行通过黄色魔方 (L525)
        print(f"[GREEN-第9步] 直行通过魔方 duty={TURN_DUTY} ratio={RATIO} time={PASS_STRAIGHT_TIME}s")
        go_straight(TURN_DUTY, RATIO)
        time.sleep(PASS_STRAIGHT_TIME)
        stop()
        time.sleep(0.2)
        print("[GREEN-第9步] 从黄色右边绕过 完成")

    # ========== 第10步: 冲线 ==========
    # 使用FINAL_RATIO补偿，全速直行冲过终点线 (L531)
    print(f"[第10步] 冲线! duty={TURN_DUTY} ratio={FINAL_RATIO} time={FINAL_STRAIGHT_TIME}s")
    go_straight(TURN_DUTY, FINAL_RATIO)
    time.sleep(FINAL_STRAIGHT_TIME)
    stop()
    print("[第10步] 冲线 完成")

    # ========== 清理资源 ==========
    print("[清理] 释放GPIO、摄像头、窗口...")
    getPic = 0
    getSpeed = 0
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("[完成] 程序结束")
