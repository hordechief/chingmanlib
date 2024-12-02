# https://github.com/Rudrabha/Wav2Lip/blob/master/inference.py

import os
import face_detection # Wav2Lip
from tqdm import tqdm
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
from ..utils.image_utils import convert_frames_to_video_opencv2,convert_frames_to_video_imageio

class FaceAlignmentExecutor():
    def __init__(self, **kwargs):
        self.face_det_batch_size = kwargs.get("face_det_batch_size",1)
        self.nosmooth = kwargs.get("nosmooth",False)
        # self.pads = kwargs.get("pads",(0,0,0,0))
        self.device = kwargs.get("device","cuda")
    
    def get_smoothened_boxes(self, boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self,images,pads = (0,0,0,0)):
        try:
            detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=self.device)
        except:
            detector = face_detection.FaceAlignment(face_detection.LandmarksType.TWO_D, flip_input=False, device=self.device)

        # batch_size = args.face_det_batch_size ########
        batch_size = self.face_det_batch_size

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        # pady1, pady2, padx1, padx2 = args.pads 
        pady1, pady2, padx1, padx2 = pads 
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        # if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
        if not self.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        # 转为RGB格式,为了跟facenet输出兼容
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), (rect[1], rect[3], rect[0], rect[2])] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results 

    # for single image
    def face_detect_ex(self, frame, pads = (0,0,0,0)):
        if type(frame) == str:
            frame = cv2.imread(frame)
        else:
            frame = frame

        results = self.face_detect([frame,], pads=pads)
        frame,cords,cords_no_pads = results[0] # 第一个脸

        return frame, list(cords), list(cords_no_pads)
    
    def gen_face_crop_drving_video(self, func_face_detect, input_video, output_video_dir, tracking_face=False, pads=(0,0,0,0)):            
        reader = imageio.get_reader(input_video)
        fps = reader.get_meta_data()['fps']
        frames = []
        try:
            cnt=0
            for frame in reader:
                if (not cnt) or tracking_face:
                    frame_face_cropped,cords, cords_no_pads = func_face_detect(frame, pads=pads)
                    y1_frame,y2_frame,x1_frame,x2_frame = list(cords).copy()
                frame = frame[y1_frame:y2_frame, x1_frame:x2_frame,:]
                # 第一个frame和其他frame BGR/RGB格式不一致的话，预测结果会变形
                frames.append(frame)
                cnt+=1
        except RuntimeError:
            pass
        reader.close()

        convert_frames_to_video_imageio(frames,output_video_dir, fps=fps)
        # IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (481, 689) to (496, 704) to ensure video compatibility with most codecs and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1 (risking incompatibility).
                
        return frames
    
    def gen_face_crop_drving_video_cv2(self, func_face_detect, input_video, output_video_dir, tracking_face=False, pads=(0,0,0,0)):
        output_image_dir = "/temp/face_crop"
        import shutil
        if os.path.exists(output_image_dir): shutil.rmtree(output_image_dir)
        os.makedirs(output_image_dir)

        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        cnt=0
        while True:
            ret, frame = cap.read() # 读入的frame是ndarray
            if not ret:
                break
            if (not cnt) or tracking_face:
                frame_face_cropped,cords, cords_no_pads = func_face_detect(frame, pads=pads)
                y1_frame,y2_frame,x1_frame,x2_frame = list(cords).copy()
            frame = frame[y1_frame:y2_frame, x1_frame:x2_frame,:]            
            frames.append(frame)
            
            if not cnt:
                plt.imshow(frame)
                plt.show()
            cnt += 1
        cap.release()
        
        # convert_frames_to_video_opencv2(frames,output_video_dir, fps=fps)
        
        # 将图像存储为视频 
        for idx,frame in enumerate(frames): cv2.imwrite('{}/{:0>4d}.jpg'.format(output_image_dir,idx),frame)
        if os.path.exists(output_video_dir): os.remove(output_video_dir)
        os.system(f"ffmpeg -framerate 24 -pattern_type glob -i '{output_image_dir}/*.jpg' -s 256x256  -c:v libx264 {output_video_dir}") 
        shutil.rmtree(output_image_dir)
            
        return frames