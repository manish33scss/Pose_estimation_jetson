import jetson.inference as JI
import jetson.utils as JU
import time


dw = 1080
dh = 720

net = JI.detectNet("ssd-mobilenet-v2", threshold = 0.5)
cam = JU.gstCamera(dw,dh, '/dev/video0')
#setup display
display = JU.glDisplay()
font = JU.cudaFont()

timeMark = time.time()
fpsFilter = 0
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while display.IsOpen():
    frame, w,h = cam.CaptureRGBA()
    classID, conf = net.Detect(frame, w,h)
    item = net.GetClassDesc(classID)
    #font.OverlayText(frame, w,h, item, 5,5,font.Magenta, font.Green)
    dt = time.time()- timeMark
    fps = 1/dt
    fpsFilter = 0.95*fpsFilter + 0.05*fps
    timeMark = time.time()
    font.OverlayText(frame,w,h, str(round(fps,1))+ 'fps' + item, 5,5,font.Magenta, font.Blue)
    
    display.RenderOnce(frame,w,h)
