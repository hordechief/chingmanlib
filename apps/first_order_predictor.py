import yaml
import torch
import cv2
import sys,os
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm import tqdm

# first-order-model
from modules.keypoint_detector import KPDetector
from modules.generator import OcclusionAwareGenerator
from sync_batchnorm import DataParallelWithCallback
from animate import normalize_kp

class FirstOrderPredictor():
    def __init__(self,
                 face_alignment_fun,
                 output='output',
                 cpu=False,
                 weight_path=None,
                 config=None,
                 relative=False,
                 adapt_scale=False,
                 find_best_frame=False,
                 best_frame=None,
                 ratio=1.0,
                 filename='result.mp4',
                 face_detector='sfd',
                 multi_person=False,
                 image_size=256,
                 face_enhancement=False,
                 batch_size=1,
                 mobile_net=False,
                 slice_size=0):
        
        self.adapt_scale = True
        self.relative = True
        self.cpu = cpu
        self.tracking_face = config["face-alignment"]["tracking_face"]
        self.pads = config["face-alignment"]["pads"]
        self.avatar = config["avatar"]["full"]
        self.outfile_animated_video = config["save_path"]["outfile_animated_video"]
        self.cropped_video_driver = config["save_path"]["outfile_cropped_video"]
        self.video_driver = os.path.join(config["DIR"]["DATA_DIR"],config["driving"]["video"]) 
        self.config = config

        self.generator, self.kp_detector = self.load_checkpoints(
           config_path=os.path.join(config["DIR"]["REPO_DIR"],'first-order-model/config/vox-256.yaml'),
           checkpoint_path=os.path.join(config["DIR"]["FOM_MODEL_DIR"],'vox-cpk.pth.tar'))
        
        FOM_REPO_DIR = os.path.join(config["DIR"]["REPO_DIR"],'first-order-model')
        if not os.path.exists(FOM_REPO_DIR):
            os.system("git clone https://github.com/navicester/first-order-model.git {}".format(config["DIR"]["REPO_DIR"]))
        if not FOM_REPO_DIR in sys.path: 
            sys.path.append(FOM_REPO_DIR)
            
        self.face_alignment_fun = face_alignment_fun

        
    # from demo import load_checkpoints  # explictly write this function here
    def load_checkpoints(self, config_path, checkpoint_path, cpu=False):

        with open(config_path) as f:
            config = yaml.load(f)

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not self.cpu:
            kp_detector.cuda()

        if self.cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)

        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])

        if not self.cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector
    
    # https://github.com/AliaksandrSiarohin/first-order-model/blob/master/demo.py

    # 流媒体处理时需要对单张图片进行处理
    # input is image
    def make_animation(
                self,
                source_image,  # 原始图片
                driving_frame, # 驱动帧
                generator, 
                kp_detector, 
                kp_driving_initial, 
                kp_source=None,
                kp_driving=None,
                relative=True, 
                adapt_movement_scale=True, 
                cpu=False):

        with torch.no_grad():
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not self.cpu:
                source = source.cuda()
            if not kp_source:
                kp_source = kp_detector(source)
            
            if not kp_driving:
                driving_frame = torch.tensor(driving_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                kp_driving= kp_detector(driving_frame)
                if not self.cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
            
            kp_norm = normalize_kp(
                        kp_source=kp_source, 
                        kp_driving=kp_driving,
                        kp_driving_initial=kp_driving_initial, 
                        use_relative_movement=relative,
                        use_relative_jacobian=relative, 
                        adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            prediction = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]

        return prediction,out

    # input is video
    def make_animation_with_video(
                self,
                source_image, 
                driving_video,
                generator, 
                kp_detector, 
                relative=True, 
                adapt_movement_scale=True, 
                cpu=False):

        predictions = []
        with torch.no_grad():
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            kp_source = kp_detector(source)
            
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                
                prediction,out = self.make_animation(
                    source_image,driving_frame,generator,kp_detector,kp_driving_initial,
                    kp_source=kp_source,kp_driving=kp_driving,relative=relative,adapt_movement_scale=adapt_movement_scale,cpu=cpu)

                # kp_norm = normalize_kp(
                #             kp_source=kp_source, 
                #             kp_driving=kp_driving,
                #             kp_driving_initial=kp_driving_initial, 
                #             use_relative_movement=relative,
                #             use_relative_jacobian=relative, 
                #             adapt_movement_scale=adapt_movement_scale)
                # out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

    # 基于图片驱动
    def driving_images(self, video_driver,source_character_face_cropped):
        from skimage.transform import resize

        i = 0
        frames = []

        cap = cv2.VideoCapture(video_driver) # 接受原始驱动视频或者经过FaceAlignment的视频
        while True:
            ret, frame = cap.read() # 读入的frame是ndarray
            if not ret:
                break

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

            # 使用第一帧取位置，偏移量都用第一帧的，如果驱动视频头像晃的太厉害就不行了
            if i == 0  or self.tracking_face:
                frame_face_crop,cords,cords_no_pads = self.face_alignment_fun(frame,pads=self.pads)
                # 这儿直接用返回的frame_face_crop有问题，基于FaceAlignment算法的检测是不带Delta的
                y1_frame,y2_frame,x1_frame,x2_frame = list(cords).copy()
                # frame = cv2.cvtColor(frame_face_crop, cv2.COLOR_BGR2RGB) # !!!!!!!!BGR和RGB格式不一样会影响驱动效果，用下面的保险
                # 或者
                frame = frame[y1_frame:y2_frame, x1_frame:x2_frame]

                plt.figure(figsize=(10,10))
                plt.subplot(1,2,1)
                plt.imshow(frame)
                plt.subplot(1,2,2)
                plt.imshow(frame_face_crop)
                plt.show()
            else:        
                frame = frame[y1_frame:y2_frame, x1_frame:x2_frame, :]

            frame = torch.from_numpy(frame) # 转换 ndarray -> Tensor
            if i == 0: # 第一帧图片会生成kp_driving_initial，它的BGR/RGB格式必须与后面帧一致
                with torch.no_grad():
                    driving_frame =  resize(frame, (256, 256))[..., :3] 
                    # driving_frame =  resize(frame, (151, 180))[..., :3]  # 原始尺寸，模型只接受256*256
                    driving_frame = torch.tensor(driving_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                    driving_frame = driving_frame.cuda()
                    kp_driving_initial = self.kp_detector(driving_frame)

            # prediction = animate_video(source_character_face_cropped, frame, kp_driving_initial, generator, kp_detector)
            # Resize image and video to 256x256
            source_image = resize(source_character_face_cropped, (256, 256))[..., :3]
            target_image = resize(frame, (256, 256))[..., :3] 
            prediction,out = self.make_animation(source_image, target_image, self.generator, self.kp_detector, relative=self.relative,
                                        adapt_movement_scale=self.adapt_scale, kp_driving_initial = kp_driving_initial)

            # prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
            frames.append(prediction*255)
            if not i:
                plt.imshow(prediction)

            i += 1
        cap.release()

        return frames
    
    def plot_avatar_image(self,source_character_face_cropped):
        plt.imshow(source_character_face_cropped)
        plt.show()

    def show_driving_video(self):
        # Driving Video

        # cropped_video_driver = "/binhe/repos/first-order-model/crop.mp4"

        # show_video(self.cropped_video_driver,height="100%",DEBUG=DEBUG)
        
        pass
        
    def run(self):
        # 被驱动图片
        source_character_face_cropped, source_character_face_cords,  source_character_face_cords_no_pads = \
                self.face_alignment_fun(os.path.join(self.config["DIR"]["DATA_DIR"], self.avatar), pads=self.pads)

        # crop之后的图片本身就是RGB格式, 下面的代码处理要RGB格式
        source_character_face_cropped = cv2.cvtColor(source_character_face_cropped, cv2.COLOR_BGR2RGB)
        
        frames = self.driving_images( self.cropped_video_driver,source_character_face_cropped)
        
        from skimage import img_as_ubyte
        imageio.mimsave(self.outfile_animated_video, [img_as_ubyte(frame/255) for frame in frames])
        
        return frames
    
    # https://github.com/AliaksandrSiarohin/first-order-model/blob/master/demo.py
    def driving_video(self, video_driver,source_character_face_cropped):
        from skimage.transform import resize
        
        source_image = source_character_face_cropped
        # source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
        source_image = resize(source_image, (256, 256))[..., :3]

        # read the video which has been processed for face detection
        reader = imageio.get_reader(video_driver)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                # driving_video.append(torch.from_numpy(im)) # 这个好像不是必须，参考git并没有，zijia代码有
                driving_video.append(im)
                # driving_video.append(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        except RuntimeError:
            pass
        reader.close()
        
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        predictions = self.make_animation_with_video(
                    source_image, 
                    driving_video, 
                    self.generator, 
                    self.kp_detector, 
                    relative=self.relative, 
                    adapt_movement_scale=self.adapt_scale, 
                    cpu=self.cpu)

        return predictions

    def run_for_video(self):
        source_character_face_cropped, source_character_face_cords,  source_character_face_cords_no_pads = \
                self.face_alignment_fun(os.path.join(self.config["DIR"]["DATA_DIR"], self.avatar), pads=self.pads)

        # crop之后的图片本身就是RGB格式, 下面的代码处理要RGB格式
        source_character_face_cropped = cv2.cvtColor(source_character_face_cropped, cv2.COLOR_BGR2RGB)
        
        predictions = self.driving_video( self.cropped_video_driver,source_character_face_cropped)
        
        from skimage import img_as_ubyte
        imageio.mimsave(self.outfile_animated_video, [img_as_ubyte(frame) for frame in predictions], fps=25)
        
        frames = [prediction*255 for prediction in predictions]

        return frames