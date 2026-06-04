import RPi.GPIO as GPIO
import time
import threading
import numpy
import matplotlib.pyplot as plt


EA, I2, I1, EB, I4, I3, LS, RS = (13, 19, 26, 16, 20, 21, 6, 12)
FREQUENCY = 50
GPIO.setmode(GPIO.BCM)
GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
GPIO.setup([LS, RS],GPIO.IN)
GPIO.output([EA, I2, EB, I3], GPIO.LOW)
GPIO.output([I1, I4], GPIO.HIGH)

pwma = GPIO.PWM(EA, FREQUENCY)
pwmb = GPIO.PWM(EB, FREQUENCY)
pwma.start(0)
pwmb.start(0)

lspeed = 0
rspeed = 0
lcounter = 0
rcounter = 0

class PID:
    """PID Controller
    """

    def __init__(self, P=80, I=0, D=0, speed=0.4, duty=26):

        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.err_pre = 0
        self.err_last = 0
        self.u = 0
        self.integral = 0
        self.ideal_speed = speed
	
    def update(self,feedback_value):
        self.err_pre = self.ideal_speed - feedback_value
        self.integral+= self.err_pre
        self.u = self.Kp*self.err_pre + self.Ki*self.integral + self.Kd*(self.err_pre-self.err_last)
        if self.u> 100:
            self.u= 100
        elif self.u < 0:
            self.u = 0
        return self.u

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain



def my_callback(channel):
    global lcounter
    global rcounter
    if (channel==LS):
        lcounter+=1
    elif(channel==RS):
        rcounter+=1
            
def getspeed():
    global rspeed
    global lspeed
    global lcounter
    global rcounter
    GPIO.add_event_detect(LS,GPIO.RISING,callback=my_callback)
    GPIO.add_event_detect(RS,GPIO.RISING,callback=my_callback)
    while True:
        rspeed=(rcounter/585.0)
        lspeed=(lcounter/585.0)
        rcounter = 0
        lcounter = 0
        time.sleep(0.1)
   

   
thread1=threading.Thread(target=getspeed)
thread1.start()



i=0
x=[]
y1=[]
y2=[]
speed = 1.9
l_origin_duty = 5
r_origin_duty = 5
pwma.start(l_origin_duty)
pwmb.start(r_origin_duty)
L_control = PID(30,0.06,20,speed,l_origin_duty)
R_control = PID(40,0.01,23,speed,r_origin_duty)

try:
    while True:
        pwma.ChangeDutyCycle(L_control.update(lspeed))
        pwmb.ChangeDutyCycle(R_control.update(rspeed))
        x.append([i])
        y1.append(lspeed)
        y2.append(rspeed)
        time.sleep(0.1)
        i+=  0.1
        print ('left: %f  right: %f lduty: %f rduty: %f'%(lspeed,rspeed,L_control.u,R_control.u))
        
except KeyboardInterrupt:
    pass
pwma.stop()
pwmb.stop()
plt.plot(x,y1,'-o')
plt.plot(x,y2,'-*')
plt.show()

GPIO.cleanup()
plt.show()
