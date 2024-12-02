import torch
import numpy as np
from tqdm import tqdm
import cv2
import subprocess,platform

import sys,os
if not os.path.exists('/binhe/repos/Wav2Lip'):
    os.system("git clone https://github.com/navicester/Wav2Lip.git /binhe/repos/")
if not '/binhe/repos/Wav2Lip' in sys.path: sys.path.append('/binhe/repos/Wav2Lip')


# Wav2Lip.models
from models import Wav2Lip
# Wav2Lib.audio
from audio import load_wav,melspectrogram 

# from Wav2Lip.inference import load_model
def _load(checkpoint_path,device):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path,device):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path,device)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

class Wav2LipExecutor():
    def __init__(self, device, face_alignment_func, config=None):
        self.box = config["face-alignment"]["box"]
        # self.static = config["face-alignment"]["static"]
        self.static = False # 固定的图片，不是动态视频
        self.fps = 25
        self.wav2lip_batch_size = 1
        self.device = device
        self.outfile_animated_video = config["save_path"]["outfile_animated_video"]
        self.outfile_final = config["save_path"]["outfile_final"]
        self.checkpoint_path = config["checkpoint_path"]["wav2lip"]
        self.img_size = config["face-alignment"]["img_size"]
        self.device=device
        self.face_alignment_func = face_alignment_func
        
    # https://github.com/Rudrabha/Wav2Lip/blob/master/inference.py
    def datagen(self, frames, mels):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                print("Using full image")
                # face_det_results = face_detect(frames, pads=args.pads) # BGR2RGB for CNN face detection
                face_det_results = self.face_alignment_func(frames,pads=(0,0,0,0))
            else:
                print("Using static image")
                # face_det_results = face_detect([frames[0]], pads=args.pads)
                face_det_results = self.face_alignment_func([frames[0]],pads=(0,0,0,0))
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        print("Data generation get {} results!".format(len(face_det_results)))

        for i, m in enumerate(mels):
            idx = 0 if self.static else i%len(frames)
            # 原始帧
            frame_to_save = frames[idx].copy()
            # 检测结果
            face, coords, coords_no_pads = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords) # !!!!!!! coords_no_pads  set pads to 0

            if len(img_batch) >= self.wav2lip_batch_size:
                # print("---------Image batch lenght is greater than audio length")
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        print("--------- image batch lengh : {}".format(len(img_batch)))
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            
    # from mylibs.wav2lip import lip_sync 代码参考Wav2Lip.inference的main函数
    # https://github.com/Rudrabha/Wav2Lip/blob/master/inference.py
    def lip_sync(self, faces=None, model=None, audio_path=None, raw_wav=None):
        mel_step_size = 16

        if self.static:
            full_frames = [faces[0]]
        else:
            full_frames = faces

        # TODO: Check if can wav = input from xunfei?
        if raw_wav:
            wav = raw_wav
        else:
            wav = load_wav(audio_path, 16000)
        mel = melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        fps = self.fps
        mel_idx_multiplier = 80./fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))
        full_frames = full_frames[:len(mel_chunks)]
        batch_size = self.wav2lip_batch_size
        # batch_size = args.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                      total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                print("frame width: {} height: {}".format(frame_h,frame_w))
                out = cv2.VideoWriter(self.outfile_animated_video,
                         cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)
            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            # for p in pred:
            #     # p = cv2.cvtColor(p, cv2.COLOR_RGB2BGR)
            #     p = p.astype(np.uint8)
            #     out.write(p)
            for p, f, c in zip(pred, frames, coords):
                f= cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                if c ==[]:
                    frame_h, frame_w = f.shape[:-1]
                    f = cv2.resize(p.astype(np.uint8), (frame_w, frame_h))
                    print("channel is empthy!")
                else:
                    y1, y2, x1, x2 = c
                    #print(c)
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2,:] = p

                out.write(np.uint8(f))
        out.release()
        command = 'ffmpeg -y -i {} -i {} -vcodec h264 -acodec aac  -strict -2 -q:v 1 {}' \
                .format(audio_path,self.outfile_animated_video, self.outfile_final)
        subprocess.call(command, shell=platform.system() != 'Windows')
        
    def run(self,frames,audio_driver):
        model = load_model(self.checkpoint_path, self.device)
        self.lip_sync(faces=frames, audio_path= audio_driver, model = model)
        # show_video(self.outfile_final,width="50%",height="100%",DEBUG=True)