import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
import cv2
import numpy as np
import PIL
from PIL import Image
import subprocess

##########################################################################
#
#                 chanel
#
##########################################################################
def imread_unchanged(img):
    if isinstance(img,str):
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if img.shape[2] > 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
    return img

def get_image_dimensions(image_path):
    # Use a library like Pillow to get the image dimensions
    from PIL import Image
    with Image.open(image_path) as image:
        return image.size

    frame = cv2.imread(image_path)
    height, width, _ = frame.shape
    return frame.shape

def get_video_dimension(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the width and height of the video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()
    
    return width,height

# --- 增加alpha通道
def add_alpha_channel(img, method = 3):
    """ 为jpg图像添加alpha通道 """
        
    if 1 == method:
        b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
        content_with_alpha = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    elif 2 == method:
        # cv2.merge
        # Create an alpha channel
        alpha_channel = np.zeros_like(img[:, :, 0])
        # Merge the content and alpha channe
        content_with_alpha = cv2.merge([img, alpha_channel]) # 4c hannel    
    elif 3 == method:
        # cv2.cvtColor
        content_with_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
    return content_with_alpha

def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加 
        y1,y2,x1,x2为叠加位置坐标值
        
        x,y - jpg coordination
        xx,yy - png coordination
    """
    
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间 !!!!!!!!!!!!!!!!!
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png  # alpha + beta  = 1
    
    # 开始叠加 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
 
    return jpg_img


def paste_image_cv2(source_image,target_image,mask_file_or_array, resize_factor=1,show_result=True):
    # import cv2

    # Step 1: Load the source image
    if isinstance(source_image,str): # opencv
        source_image = cv2.imread(source_image, cv2.IMREAD_UNCHANGED) 
    else:
        source_image = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    # print(source_image.shape) # (620, 930, 3)
    source_height, source_width = source_image.shape[:2] 

    # Step 2: Load the mask
    if  isinstance(mask_file_or_array,str):
        # Step 2B: Load the mask from stored file
        mask = cv2.imread(mask_file_or_array, cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask_file_or_array,np.ndarray):
        mask = mask_file_or_array
    elif not mask_file_or_array:
        print("mask file missing")
        return None
    print("mask shape:",mask.shape)

    # Step 3: Extract the masked region and add alpha channel
    # extract imaage
    extracted_image = cv2.bitwise_and(source_image, source_image, mask=mask)
    # add alpha channel
    extracted_image_with_alpha = add_alpha_channel(extracted_image)
    # set the value of alpha channel to mask
    extracted_image_with_alpha[:, :, 3] = mask
    # cv2.imwrite("/binhe/output/segmented_cat.png", extracted_image_with_alpha)
    print("extracted image with alpha shape: ",extracted_image_with_alpha.shape)

    # Step 4: Resize the extracted image (resize can be done before or after extraction)
    new_width, new_height = int(source_width*resize_factor),int(source_height*resize_factor)
    print("new_width, new_height: ", (new_width, new_height))
    extracted_image_with_alpha_resized = cv2.resize(extracted_image_with_alpha, (new_width, new_height))  # Replace new_width and new_height with the desired dimensions
    # cv2.imwrite("/binhe/output/segmented_cat_resized.png", extracted_image_with_alpha_resized)

    # Step 5: Load the target image
    target_image = cv2.imread(target_image, cv2.IMREAD_UNCHANGED)
    target_height, target_width = target_image.shape[:2]
    print("target image shape: ",target_image.shape)

    # Step 6: Find the region of interest in the target image (e.g., coordinates or location)
    x,y,w,h = 0,0,new_width,new_height

    # Step 7: Paste the resized image onto the target image
    target_image_with_alpha = target_image
    if target_image_with_alpha.shape[2] == 3:
        target_image_with_alpha = add_alpha_channel(target_image)
    # target_image_with_alpha[y:y+h, x:x+w] = extracted_image_with_alpha_resized  # ！！！这句会把黑色背景带进来，改用下面的方法
    mask = cv2.resize(mask,(new_width,new_height))
    alpha_source =  mask / 255.0 
    alpha_target = 1 - alpha_source 
    print("alpha source shape: ",alpha_source.shape)
    for c in range(0,3):
        target_image_with_alpha[y:y+h, x:x+w, c] = \
                ((alpha_target * target_image_with_alpha[y:y+h, x:x+w, c]) + 
                 (alpha_source * extracted_image_with_alpha_resized[0:h,0:w,c]))


    # Step 8: Save or display the modified target image
    cv2.imwrite('/binhe/output/output_image.jpg', target_image_with_alpha)

    if show_result:
        plt.figure(figsize=(12,12))
        plt.subplot(1,3,1)
        plt.imshow(source_image)
        plt.subplot(1,3,2)
        plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR))
        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor(target_image_with_alpha, cv2.COLOR_BGRA2RGBA))
        
    return target_image_with_alpha

def paste_image_CV2_alt(src_img,target_img,mask,x=10,y=10,delta=0,resize=1,is_show=False):
   
    # 如何输入的是4通道的png图片，则mask是可选的
    # 如何输入的是jpg图片，那么必须同时输入mask文件
    
    # Step 1: Load the source image (PNG with transparent background)    
    if isinstance(src_img,str):
        source_image = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
    else:
        source_image = src_img
    if source_image.shape[2] < 4:
        source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGBA)
    print("source image shape",source_image.shape)    

    # Step 3: Extract the alpha channel from the source image
    alpha_channel = source_image[:, :, 3]

    # Step 4: Extract the RGB channels from the source image
    source_image = source_image[:, :, :3]
    print("after extraction - alpha_channel.shape:",alpha_channel.shape," source_image.shape:",source_image.shape)

    # Step 5: Resize the source image to match the desired size if needed
    # You can use cv2.resize() to resize the source image
    new_height = int(source_image.shape[0]*resize)
    new_width = int(source_image.shape[1]*resize)
    alpha_channel = cv2.resize(alpha_channel,(new_width,new_height))
    source_image = cv2.resize(source_image, (new_width,new_height))
    print("resized source image size:", source_image.shape)

    # Step 6: Create a mask by thresholding the alpha channel
    # 如何source image传进来的是png，可以自己提取，如果传进来的source image是jpg，需要外面输入mask
    if  isinstance(mask,str):
        # 方法2：通过读取mask png文件获取mask
        # mask = cv2.imread('/binhe/output/mask_cat.png', cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask,np.ndarray):
        # 方法3：根据segementaton返回的mask处理，这个mask里的每个值是True或者False。或者可以认为这是一个可选的特别的mask
        # mask = mask*255 # 传进来之前处理好，不在这儿转换
        # mask = mask.astype('uint8')  
        pass
    elif not mask:
        if src_img.endwith(".jpg"):
            print("mask file missing when the input source image is jpg file")
            return None
        elif src_img.endwith(".png"):    
            # 方法1：通过alpha通道提取
            _, mask = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
    
    mask = cv2.resize(mask,(new_width,new_height))
    
    # Step 7: Invert the mask
    mask_inv = cv2.bitwise_not(mask)
    print("mask inv shape:",mask_inv.shape)
    # plt.imshow(mask_inv)

    # Step 2: Load the target image
    target_image = cv2.imread(target_img)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    print("target image shape",target_image.shape)
    
    # Step 8: Bitwise AND between the target image and the inverted mask
    target_image_pasted = target_image[y:y+new_height,x:x+new_width]
    print("target_image_pasted.shape: ",target_image_pasted.shape)
    background = cv2.bitwise_and(target_image_pasted, target_image_pasted, mask=mask_inv) # 1 - mask

    # Step 9: Bitwise AND between the source image and the mask
    foreground = cv2.bitwise_and(source_image, source_image, mask=mask)
    print("foreground shape:",foreground.shape, " background shape: ",background.shape) # mask

    # Step 10: Combine the foreground and background using bitwise OR
    #h,w=new_height,new_width
    result = cv2.bitwise_or(foreground, background)

    # Step 11: Paste the result onto the target image    
    target_image[y:y+new_height, x:x+new_width] = result

    # Step 12: Save the modified target image
    # cv2.imwrite('/binhe/output/output_image.jpg', target_image)

    if is_show:
        plt.figure(figsize=(12,12))
        plt.subplot(1,3,1)
        plt.imshow(source_image)
        plt.subplot(1,3,2)
        plt.imshow(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        plt.subplot(1,3,3)
        plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGRA2RGBA))
        #plt.imshow(target_image)
        
        
    return target_image

def paste_image_PIL(src_img,target_img,mask=None,coords=(10,10),pads=(0,0,0,0),resize=1,is_show=False):
    # from PIL import Image
    # import numpy as np

    # Opening the secondary image (overlay image)
    if isinstance(src_img,str):
        pillow_image = PIL.Image.open(src_img)
    elif isinstance(src_img,np.ndarray):  # opencv
        pillow_image = PIL.Image.fromarray(src_img) # Pillow
    elif isinstance(src_img,PIL.Image.Image):
        pillow_image = src_img
    else:
        print("unknown input source image type")

    width, height = pillow_image.size
    new_width, new_height = int(width*resize), int(height*resize)
    if not 1==resize:
        pillow_image = pillow_image.resize((new_width, new_height))
    
    # Opening the primary image (used in background)
    if isinstance(target_img,str):
        target_image = PIL.Image.open(target_img)
    elif isinstance(target_img,np.ndarray):  # opencv
        target_image = PIL.Image.fromarray(target_img) # Pillow
    elif isinstance(target_img,PIL.Image.Image):
        target_image = target_img
    else:
        print("unknown input target image type")    
    target_image_origin = target_image.copy()
    
    # 这段代码待验证
    if not mask:
        mask = pillow_image
        
    # starting at coordinates (0, 0)
    # (coords[0]-pads[2],coords[1]-pads[0])
    target_image.paste(pillow_image, coords , mask = mask)
    
    # Displaying the image
    if is_show:
        #target_image.show()
        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1)
        plt.title("source image",fontsize=8)
        plt.imshow(np.asarray(pillow_image))
        plt.subplot(1,3,2)
        plt.title('target image origin',fontsize=8)
        plt.imshow(np.asarray(target_image_origin))
        plt.subplot(1,3,3)
        plt.title('target image pasted',fontsize=8)
        plt.imshow(np.asarray(target_image))
        plt.show()
        
    return target_image

    
##########################################################################
#
#                 Remove Background
#
##########################################################################
from rembg import remove

# 目前google drive下载有问题，
# 手动下载https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab ，
# 把u2net.onnx放到/root/.u2net/u2net.onnx

# Downloading data from 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx' to file '/root/.u2net/u2net.onnx'.

'''
if not os.path.exists("/root/.u2net/u2net.onnx"):
    os.makedirs("~/.u2net")
    os.system("wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -p ~/.u2net/")
'''

def RemoveBackground(input_path:str,output_path:str):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)


def RemoveBackgroundErode(source_image, target_image, kernel=((2,2),2), show_result=False): 
    if show_result: print("Start RemoveBackgroundErode ------------------")
    RemoveBackground(source_image,target_image)
    
    # target image - after remove
    target_image_rmbg = cv2.imread(target_image, cv2.IMREAD_UNCHANGED)
    if show_result: print("image shape after rmbg: ", target_image_rmbg.shape)
    alpha_channel = target_image_rmbg[:, :, 3]
    
    # get the alpha channel with erosion
    if not kernel[0] and not kernel[1]:
        kernel_size,iterations = kernel
    #     kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Change kernel size as needed
    #     kernel = np.ones((1, 5), np.uint8)
        kernel = np.ones(kernel_size, np.uint8)
        alpha_channel_eroded = cv2.erode(alpha_channel.astype('uint8'), kernel, iterations=iterations)
    else:
        alpha_channel_eroded = alpha_channel
    
    # final content image with erosion
    source_image_array = cv2.imread(source_image, cv2.IMREAD_UNCHANGED)
    content_image = cv2.bitwise_and(source_image_array, source_image_array, mask=alpha_channel_eroded)
    if show_result: print("content image shape: ", content_image.shape)
    if content_image.shape[2] == 3:
        content_image = add_alpha_channel(content_image)
        content_image[:, :, 3] = alpha_channel_eroded#.astype('uint8')*255
        
    # write the content to target image
    cv2.imwrite(target_image,content_image)

    if show_result:
        plt.figure(figsize=(8,8))
        plt.subplot(1,3,1)
        plt.imshow(alpha_channel)
        plt.title("alpha channel",fontsize=8)
        plt.subplot(1,3,2)
        plt.imshow(alpha_channel_eroded)
        plt.title("alpha channel eroded",fontsize=8)
        plt.subplot(1,3,3)
        plt.imshow(content_image)
        plt.title("content image extracted",fontsize=8)
        plt.show()
        
    return content_image,alpha_channel_eroded

def MyRemoveBackground(input_path:str,output_path:str,kernel = ((2,2),2),show_result=False):
    if show_result: print("Start MyRemoveBackground ------------------")

    # Load the image with the green screen background
    image = cv2.imread(input_path)
    # hsv = cv2.cvtColor(opencv, cv2.COLOR_RGB2HSV)
    
    # Define the lower and upper bounds for the green color (in BGR format)
    lower_green = np.array([35, 80, 80])  # Adjust these values as needed
    upper_green = np.array([85, 255, 255])  # Adjust these values as needed
    lower_green = np.array([79, 174, 75])  # Adjust these values as needed
    upper_green = np.array([81, 177, 77])  # Adjust these values as needed
    lower_green = np.array([60, 160, 60])  # Adjust these values as needed
    upper_green = np.array([80, 180, 80])  # Adjust these values as needed
    # 
#     lower_green = np.array([0, 255, 0])  # Adjust these values as needed
#     upper_green = np.array([0, 255, 0])  # Adjust these values as needed
#     lower_green = np.array([0, 240, 0])  # Adjust these values as needed
#     upper_green = np.array([10, 255, 10])  # Adjust these values as needed

    # 80,176,76

    # Create a mask to identify the green color
    mask = cv2.inRange(image, lower_green, upper_green)

    # Invert the mask (to select non-green pixels)
    mask_inv = cv2.bitwise_not(mask)

    # Erode the mask to remove edge artifacts
#     kernel_size = 5
#     kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Change kernel size as needed
#     mask_inv = cv2.erode(mask_inv, kernel, iterations=2)
    kernel_size,iterations = kernel
    kernel = np.ones(kernel_size, np.uint8) 
    mask_inv = cv2.erode(mask_inv, kernel, iterations=iterations)

    # Apply the mask to the original image, making the green screen transparent
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)

    # Create an alpha channel for transparency
    #alpha = np.ones_like(foreground[:, :, 0]) * 255
    alpha = np.zeros_like(foreground[:, :, 0])

    # # todo 拆分为3通道
    # b, g, r = cv2.split(foreground)

    # # todo 合成四通道
    # bgra = cv2.merge([b, g, r, mask_inv])

    # Combine the foreground and alpha channel into a 4-channel image
    bgra = cv2.merge((foreground, alpha))
    print(bgra.shape)
    
    bgra[:, :, 3] = mask_inv

    # Save the result as a PNG image with transparency
    cv2.imwrite(output_path, bgra)   
    
    if show_result:    
        plt.figure(figsize=(10,10))
        plt.subplot(1,4,1)
        plt.title("origin image",fontsize=8)
        plt.imshow(image)
        plt.subplot(1,4,2)
        plt.title('mask',fontsize=8)
        plt.imshow(mask)
        plt.subplot(1,4,3)
        plt.title('mask inverse',fontsize=8)
        plt.imshow(mask_inv)
        plt.subplot(1,4,4)
        plt.title('bgra',fontsize=8)
        # Alternatively, if you want to display the image with transparency
        plt.imshow(bgra)
        plt.show()
        
    return bgra,mask_inv



###########################################################################################
#
#               Image Streaming
#
#
###########################################################################################
def get_first_frame(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return None

    # Read the first frame
    success, frame = video.read()

    # Check if the frame was read successfully
    if not success:
        print("Error reading video frame")
        return None

    # Release the video file
    video.release()

    return frame

def convert_images_to_video(image_folder, audio_driver, output_file, fps=25, image_type="jpg"):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Sort the image files in ascending order
    image_files.sort()

    # Determine the output video's width and height based on the first image
    first_image = os.path.join(image_folder, image_files[0])
    width, height = get_image_dimensions(first_image)
    
    print(f"width {width},height {height}, fps {fps}")

    # Generate the ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg', 
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', os.path.join(image_folder, f'*.{image_type}'),  # Input image files
        '-i',audio_driver,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        '-strict','-2',
        output_file,  # Output file path
    ]
        
    # Run the ffmpeg command
    subprocess.run(ffmpeg_cmd, check=True)
    
def convert_images_to_video_ffmpeg(image_folder, output_video, framerate_input = None,fps=24, audio_driver=None, image_type="jpg"): 
    import subprocess    

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
 
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Sort the image files in ascending order
    image_files.sort()
    
    # Determine the output video's width and height based on the first image
    first_image = os.path.join(image_folder, image_files[0])
    width, height,_ = cv2.imread(first_image).shape
    print(width,height,fps)

    # Generate the ffmpeg command
    ffmpeg_cmd = [
        'ffmpeg', 
        '-y', 
        '-r', str(fps), 
        '-i', os.path.join(image_folder, '%*.{}'.format(image_type)), 
        '-vcodec', 'mpeg4', 
        output_video
    ]

    ffmpeg_cmd = [
        'ffmpeg', 
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', os.path.join(image_folder, '*.{}'.format(image_type)),  # Input image files
#         '-i',audio_driver,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        output_video  # Output file path
    ]

    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Overwrite output file if it exists
        '-f', 'image2',  # Input format
        '-r', str(fps),  # Frames per second
        '-s', '{}x{}'.format(width, height),  # Input image size
        '-i', os.path.join(image_folder, '%d.{}'.format(image_type)),  # Input image files
        '-vcodec', 'libx264',  # Output video codec
        '-crf', '18',  # Constant Rate Factor (lower value = higher quality)
        '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
        output_video  # Output file path
    ]
    
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', os.path.join(image_folder, '*.{}'.format(image_type)),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]

    # os.system("ffmpeg -y -framerate 25 -pattern_type glob -i '{}/*.jpg'  -c:v libx264 {}" \.format(image_folder,output_video))

    # Verified Command
    cmd = [
            'ffmpeg', 
            '-y', # overwrite
            '-framerate', str(framerate_input), # specify this frame rate for the input.
            '-pattern_type', 'glob', 
            '-i', f'{image_folder}/*.{image_type}', 
            '-i', f'{audio_driver}', 
            '-c:v', 'libx264',  #  video codec -  libx264 (H.264) and libx265 (H.265/HEVC). = -c:v  (short for -codec:v) 
            "-c:a", "aac", # audio codec - aac, mp3, and pcm_s16le.
            '-pix_fmt', 'yuv420p', # pixel format for video streams
            "-strict", "experimental", # the "strict" flag is used to control the strictness level of certain codec compliance
            "-r", str(fps), #  specifies the output frame rate for the video
            output_video
     ]
        
    cmd = ['ffmpeg','-y']
    # input
    if framerate_input:cmd.extend(['-framerate', str(framerate_input),])
    cmd.extend(['-pattern_type', 'glob',])
    cmd.extend(['-i', f'{image_folder}/*.{image_type}',])
    if audio_driver:cmd.extend(['-i', f'{audio_driver}',])
    cmd.extend(['-c:v', 'libx264',])
    if audio_driver:cmd.extend(["-c:a", "aac",])
    cmd.extend(['-pix_fmt', 'yuv420p',"-strict", "experimental","-r", str(fps),])
    cmd.extend([output_video,])
        
    print(cmd)

    subprocess.call(cmd)

    return output_video

# opencv实现
def convert_frames_to_video_opencv2(frames, output_video, fps=24): 

    # Determine the size of the images based on the first image
    height, width, layers = frames[0].shape

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on the file extension
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    # Release the VideoWriter and close the video file
    video_writer.release()
    
    return output_video


def convert_images_to_video_opencv2(image_folder, output_video, fps=24): 
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg') or img.endswith('.png')]  # Update the file extension if necessary

    images = sorted(images)
    #images.sort(key=lambda x: int(x.split('image')[1].split('.jpg')[0]))

    frames = [cv2.imread(os.path.join(image_folder, image)) for image in images]

    return convert_frames_to_video_opencv2(frames, output_video, fps)

def convert_frames_to_video_imageio(frames,output_video_path, fps=25):
    from skimage import img_as_ubyte
    import imageio
    imageio.mimsave(output_video_path, [img_as_ubyte(frame/255) for frame in frames], fps=fps)


def gen_video_from_numpy_frames(frames,output_video_path,temp_frame_save_dir=None, fps=25):
    import shutil
    if os.path.exists(output_video_path):os.remove(output_video_path)
    if not temp_frame_save_dir: temp_frame_save_dir = "/tmp/avatar-animate"
    if os.path.exists(temp_frame_save_dir): shutil.rmtree(temp_frame_save_dir)
    os.makedirs(temp_frame_save_dir)
    
    # 将图片保存为视频
    print(f"temp images folder: {temp_frame_save_dir} \ngenerated video file: {output_video_path}")

    # 将帧保存为图片
    for i,_ in enumerate(frames):              
        im = PIL.Image.fromarray(_.astype('uint8'))
        num_channels = im.mode
        mode = "RGB"
        if num_channels == 4: mode="RGBA"
        im = im.convert(mode) # JPG RGB就可以了，PNG要RGBA
        im.save('{}/{:0>4d}.jpg'.format(temp_frame_save_dir,i))
        # print(im.size)

    # 输出jpg
    os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i '{temp_frame_save_dir}/*.jpg'  -c:v libx264 {output_video_path}")

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Image Uitlity')
    par.add_argument('-m', '--image', type=str, default=source, 
                     help='image path.')