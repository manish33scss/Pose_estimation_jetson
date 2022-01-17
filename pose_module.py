# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 23:06:48 2022

@author: ASUS
"""

import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import numpy as np

class Pose:
    def __init__(self, model = "resnet"):
        
        with open('script/human_pose.json', 'r') as f:
            self.human_pose = json.load(f)
        
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_pose)

        self.num_parts = len(self.human_pose['keypoints'])
        self.num_links = len(self.human_pose['skeleton'])
        self.keypoints = list()
        
        if not os.path.exists('weight'):
            os.makedirs('weight')
        if not os.path.exists('model'):
            os.makedirs('model')
        
        if model == 'resnet':
            print('------ model = resnet--------')
            self.MODEL_WEIGHTS = 'weight/resnet18_baseline_att_224x224_A_epoch_249.pth'
            self.OPTIMIZED_MODEL = 'model/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
            self.model = trt_pose.models.resnet18_baseline_att(self.num_parts, 2 * self.num_links).cuda().eval()
            self.WIDTH = 224
            self.HEIGHT = 224

        else:    
            print('------ model = densenet--------')
            self.MODEL_WEIGHTS = 'weight/densenet121_baseline_att_256x256_B_epoch_160.pth'
            self.OPTIMIZED_MODEL = 'weight/densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
            self.self.model = trt_pose.models.densenet121_baseline_att(self.num_parts, 2 * self.num_links).cuda().eval()
            self.WIDTH = 256
            self.HEIGHT = 256

        self.data = torch.zeros((1, 3, self.HEIGHT, self.WIDTH)).cuda()
        if os.path.exists(self.OPTIMIZED_MODEL) == False:
            self.model.load_state_dict(torch.load(self.MODEL_WEIGHTS))
            self.model_trt = torch2trt.torch2trt(self.model, [data], fp16_mode=True, max_workspace_size=1<<25)
            torch.save(self.model_trt.state_dict(), self.OPTIMIZED_MODEL)

        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(self.OPTIMIZED_MODEL))
        
        self.t0 = time.time()
        torch.cuda.current_stream().synchronize()
        for i in range(50):
            y = self.model_trt(self.data)
        torch.cuda.current_stream().synchronize()
        self.t1 = time.time()
        
        print(50.0 / (self.t1 - self.t0))
        
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')
        self.parse_objects = ParseObjects(self.topology)
        self.draw_objects = DrawObjects(self.topology)
        self.X_compress = 640.0 / self.WIDTH * 1.0
        self.Y_compress = 480.0 / self.HEIGHT * 1.0
                
    def get_keypoint(self, humans, hnum, peaks):
            #check invalid human index
            kpoint = []
            human = humans[0][hnum]
            C = human.shape[0]
            for j in range(C):
                k = int(human[j])
                if k >= 0:
                    peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
                    peak = (j, float(peak[0]), float(peak[1]))
                    kpoint.append(peak)
                    #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
                else:    
                    peak = (j, None, None)
                    kpoint.append(peak)
                    #print('index:%d : None %d'%(j, k) )
            return kpoint
    

        
    def preprocess(self, image):
            #global device
            #device = torch.device('cuda')
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
            image = transforms.functional.to_tensor(image).to(self.device)
            image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            return image[None, ...]

    def execute(self, img,  t, frame_count, black_screen = False, save_json =True):
            src = img.copy()
            w,h,c = src.shape
            
            img = cv2.resize(img, dsize=(self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_AREA)
            color = (0, 255, 0)
            data = self.preprocess(img)
            self.cmap, self.paf = self.model_trt(data)
            self.cmap, self.paf = self.cmap.detach().cpu(), self.paf.detach().cpu()
            self.counts, self.objects, self.peaks = self.parse_objects(self.cmap, self.paf)#, cmap_threshold=0.15, link_threshold=0.15)
            fps = 1.0 / (time.time() - t)
            for i in range(self.counts[0]):
                 keypoints = self.get_keypoint(self.objects, i, self.peaks)
                 #print(keypoints)
                 self.keypoints.append( {
                        "frame number" : frame_count,
                        "keypoints" : keypoints

                 })

                #  for j in range(len(keypoints)):
                #      if keypoints[j][1]:
                #          x = round(keypoints[j][2] * self.WIDTH * self.X_compress)
                #          y = round(keypoints[j][1] * self.HEIGHT * self.Y_compress)
                #          print(x,y)
            #             cv2.circle(src, (x, y), 3, color, 2)
            #             cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            #             cv2.circle(src, (x, y), 3, color, 2)
            print("FPS:%f "%(fps))

            if black_screen:
                blank_frame = np.zeros(shape = [w,h,c], dtype = np.uint8)
                self.draw_objects(blank_frame, self.counts, self.objects, self.peaks)
                cv2.putText(blank_frame , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.imshow("frame", blank_frame)
                out_video.write(blank_frame)
            else:
                self.draw_objects(src, self.counts, self.objects, self.peaks)
                cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.imshow("frame", src)
                out_video.write(src)
            
            if save_json:

                with open("result.json", 'w') as f:
                    f.write(json.dumps(self.keypoints))

            
            
            out_video.write(blank_frame)
                
        
        
        
        
        


if __name__ == "__main__":
    
    path = ""
    cap = cv2.VideoCapture("data/pose.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    #ret_val, img = cap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
    count = 0
    pose = Pose()
    #WIDTH = 640.0 
    #HEIGHT = 480.0
    
    if cap is None:
        print("Camera Open Error")
        sys.exit(0)
    

    
    while True:
        t = time.time()
        ret_val, img = cap.read()
        if ret_val == False:
            print("Camera read Error")
            break
    
        #img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        count += 1
        pose.execute(img,t, count, black_screen = False, save_json = True)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    
    cv2.destroyAllWindows()
    out_video.release()
    cap.release()

'''[(0, 0.7013946771621704, 0.5816441774368286), (1, 0.6546362638473511, 0.6100377440452576), (2, 0.6619329452514648, 0.5467584729194641), (3, 0.6668516397476196, 0.652735710144043), (4, 0.672402024269104, 0.5042949914932251), (5, 0.8444611430168152, 0.7293646931648254), (6, 0.8324268460273743, 0.4358384907245636), (7, None, None), (8, None, None), (9, None, None), (10, None, None), (11, None, None), (12, None, None), (13, None, None),
\ (14, None, None), (15, None, None), (16, None, None), (17, 0.8426513075828552, 0.5834174752235413)] '''