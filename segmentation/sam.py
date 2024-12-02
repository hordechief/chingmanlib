from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry,SamPredictor

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
import cv2
import numpy as np


class SAMExecutor():
    def __init__(self, **config):
        # Choose a model type and a checkpoint
        model_type = config.get("model_type", "vit_h")
        checkpoint_path = config.get("checkpoint_path", "./models/sam_vit_h_4b8939.pth")
        # checkpoint = "./models/sam_vit_b_01ec64.pth"
        # checkpoint = "./models/sam_vit_l_0b3195.pth"

        # Create a SAM model and a predictor
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
    
    def save_mask_contour(self, mask, object_image_with_alpha, show_result=False):
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_x, mask_y, mask_w, mask_h = cv2.boundingRect(contours[0])
        # print(mask_x,mask_y,mask_w,mask_h)

        cropped_image = object_image_with_alpha[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w]

        if show_result:
            plt.imshow(cropped_image)
            
        return cropped_image
        
    def run(
        self,
        input_path:str,
        output_path:str,
        prompt, # (np.array([[100, 100]]),np.array([1]),np.array([0,0,100,100])) 
        mask_index=1, 
        kernel = ((2,2),2), 
        transparent_background_color=True, 
        show_result=False, 
        **kwargs):
            
        # Load an image
        image = cv2.imread(input_path)
        if image.shape[2] > 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image.shape)

        # Set the image for the predictor
        self.predictor.set_image(image)
        
        input_point = prompt.get("point_coords",None)
        input_label = prompt.get("point_labels",None)
        input_box = prompt.get("box",None)
        
        if np.array(input_box).size:
            multimask_output = False
            mask_index = 0
        else:
            multimask_output = True
            mask_index = mask_index
            
        prompt.update({
            "multimask_output":multimask_output,
        })

        # predict
        masks, scores, logits = self.predictor.predict(
            **prompt,
        )
        # print(masks[2].shape)

        if show_result:
            plt.figure(figsize=(10,10))
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.subplot(1,3,i+1)
                plt.imshow(image)
                self.show_mask(mask, plt.gca())
                self.show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=12)
            plt.axis('off')
            plt.show()  

        score = mask_index #np.argmax(scores) #2 # 分数最高的不一定最准确，经验是2，面积最大
        logit = logits[score, :, :] 
        mask = masks[score, :, :]#.astype('uint8')*255
        # print(score,mask)
        if show_result:
            print(scores)

        # Test code
        # height, width, _ = image.shape
        # mask = np.ones((height, width), dtype=np.uint8) * 255
        # center_x, center_y = width // 2, height // 2
        # rect_width, rect_height = width // 2, height // 2
        # cv2.rectangle(mask, (center_x - rect_width // 2, center_y - rect_height // 2), 
        #           (center_x + rect_width // 2, center_y + rect_height // 2), 0, thickness=cv2.FILLED)

        # kernel_size = 2
        # iterations = 1
        # kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Change kernel size as needed
        # kernel = np.ones((1, 5), np.uint8)
        kernel_size,iterations = kernel
        kernel = np.ones(kernel_size, np.uint8) 

        # extract the content
        _mask = mask*255
        mask_eroded = cv2.erode(_mask.astype('uint8'), kernel, iterations=iterations)

        mask = mask_eroded
        mask_inverse =  cv2.bitwise_not(mask) # 这个地方mask的值必须是乘过255的，否则虽然显示没问题，但not的结果不正确

        content_image = cv2.bitwise_and(image, image, mask=mask)
        alpha_channel = np.zeros_like(content_image[:, :, 0]) 
        # print(alpha_channel.shape)

        # Merge the content and alpha channel
        object_image_with_alpha = cv2.merge([content_image, alpha_channel]) # 4 channel
        object_image_with_alpha[:, :, 3] = mask
        
        object_image_with_alpha = cv2.cvtColor(object_image_with_alpha, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(output_path, object_image_with_alpha)

        if show_result:
            # Display the reuslt
            plt.figure(figsize=(10,10))
            plt.subplot(1,5,1)
            plt.imshow(image)
            plt.title("raw image", fontsize=8)
            plt.subplot(1,5,2)
            plt.title("mask", fontsize=8)
            plt.imshow(mask)
            plt.subplot(1,5,3)
            plt.title("mask eroded", fontsize=8)
            plt.imshow(mask_eroded)
            plt.subplot(1,5,4)
            plt.title("masked content image", fontsize=8)
            plt.imshow(content_image)
            plt.subplot(1,5,5)
            plt.title("final image with alpha", fontsize=8)
            plt.imshow(object_image_with_alpha)
            plt.show()

        return mask,mask_inverse 
        
    def plot_mask(self, image, verbose=False):
        masks = self.mask_generator.generate(image)
        if verbose:
            # print(masks[0].keys())
            # print(masks)

            plt.figure(figsize=(25,5))
            for i,mask in enumerate(masks[0:5]):
                plt.subplot(1,5,i+1)
                plt.imshow(mask['segmentation'].astype('uint8'))
            plt.show()

    def predict_one_image(self,path_or_array,prompt):
        # Load an image    
        # print(type(path_or_array))
        if isinstance(path_or_array,str):
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = path_or_array
        #print(image.shape)

        self.predictor.set_image(image)

        input_point = prompt.get("point_coords",None)
        input_label = prompt.get("point_labels",None)
        input_box = prompt.get("box",None)
        
        if np.array(input_box).size:
            multimask_output = False
            mask_index = 0
        else:
            multimask_output = True
            mask_index = 2
            
        prompt.update({
            "multimask_output":multimask_output,
        })
            
        masks, scores, logits = self.predictor.predict(
            **prompt
        )

        return image,masks,scores,logits,mask_index

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
