import os,cv2
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import glob

'''
import sys
sys.path.append('/binhe/ml-ex')

from importlib import reload # python3.2 and 3.3
# from imp import reload # python3.4+
import mylibs #as mylibs
try:
    reload(mylibs.image_processing)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
from mylibs.image_processing import imShow,imShowPlt,show_video
'''

# %matplotlib inline

###########################################################################################
#
#               Image(s) & Video Display
#
#
###########################################################################################

def imShow(path:str, scale=3, size_inches=(18, 10)):
    # 透明背景在 OpenCV 中打开时显示为黑色是因为 OpenCV 默认使用 <BGR> 颜色空间。
    # 如果你希望保留透明度信息并在 OpenCV 中正确显示透明背景，你需要使用 <RGBA> 颜色空间。

    # OpenCV提供的图像读取函数cv2.imread()，在默认情况下读取png图像会自动忽略Alpha通道，即png图像直接变为jpg图像。

    # import cv2
    try:
        # Load the image using the imread function
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED) # 正常读入图像，并保留其通道数不变，png图像为4通道，jpg图像为3通道
        height, width = image.shape[:2]
    except:
        print("imShow: fail to open file",path)
        return
        
    resized_image = cv2.resize(image,(scale*width, scale*height), interpolation = cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(*size_inches) #(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)) # 默认读进来的是BGR，matplot处理的是RGB图片
    plt.show()

def imShowMPIMG(img_path:str):
    # import matplotlib.image as mpimg    
    img = mpimg.imread(img_path)
    plt.imshow(img)

def imShowTF(path, image_size=(150, 150)):
    img = tf.keras.preprocessing.image.load_img(path, target_size=image_size) 
    plt.imshow(img)

def show_image_PIL(image:str):
    img = Image.open(image)
    if image.endswith(".png"):
        img = img.convert("RGBA")
    img.show()
    
def show_video(video_path,width="50%",height="50%",DEBUG=False):
    if not DEBUG: return
    from base64 import b64encode
    from IPython.display import HTML
    mp4 = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(f"""
    <video width={width} height={height} controls>
          <source src="{data_url}" type="video/mp4">
    </video>""")

# not work
def show_video_IPython(video_path:str):
    from IPython.display import Video
    Video(video_path,embed=True)
    
def show_video_CV2(video_path:str,small:int=2):
    if not os.path.exists(video_path):
        print("video not exist")
    video = cv2.VideoCapture(video_path)
    current_time = 0
    while(True):
        try:
            clear_output(wait=True)
            ret, frame = video.read()
            if not ret:
                break
            lines, columns, _ = frame.shape
            #########do img preprocess##########
            
            # 画出一个框
            #     cv2.rectangle(img, (500, 300), (800, 400), (0, 0, 255), 5, 1, 0)
             # 上下翻转
             # img= cv2.flip(img, 0)
            
            ###################################
            
            if current_time == 0:
                current_time = time.time()
            else:
                last_time = current_time
                current_time = time.time()
                fps = 1. / (current_time - last_time)
                text = "FPS: %d" % int(fps)
                cv2.putText(frame, text , (0,100), cv2.FONT_HERSHEY_TRIPLEX, 3.65, (255, 0, 0), 2)
                
          #     img = cv2.resize(img,(1080,1080))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (int(columns / small), int(lines / small)))

            img = Image.fromarray(frame)

            display(img)
            # 控制帧率
            time.sleep(0.02)
        except KeyboardInterrupt:
            video.release()

def show_images_from_directory(dir_path,image_type="jpg",n_row=3,n_column=3, figsize=(8, 8)):
#     import os, glob
#     import matplotlib.pyplot as plt
#     from PIL import Image

    # Get a list of all image files in the directory
    image_list = glob.glob(os.path.join(dir_path, '*.'+image_type))

    # Create a figure with 3x3 subplots
    fig, axs = plt.subplots(nrows=n_row, ncols=n_column, figsize=figsize)

    # Loop over the image paths and display them on the subplots
    for i, ax in enumerate(axs.flatten()):
        # Load the image and display it on the current subplot
        image = Image.open(image_list[i])
        ax.imshow(image)
        ax.axis('off')

        # Stop the loop if we have displayed 9 images
        if i == n_row*n_column-1:
            break

    # Display the figure
    plt.show()

def show_images_from_directory_alt1(image_folder_path,n_row=3,n_column=3, figsize=(16, 32)):
    # Initialize a figure with a 9x1 grid of subplots
    fig, axs = plt.subplots(n_row, n_row, figsize=figsize)

    # Loop through the images in the folder and plot them on the subplots
    for i, filename in enumerate(os.listdir(image_folder_path)):
        if i >= n_column*n_row:
            break
        # Load the image
        img_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(img_path)

        # Convert the color from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Plot the image on the corresponding subplot
        row = i // n_column
        col = i % n_column
        axs[row,col].imshow(img)
        axs[row,col].axis('off')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # Display the figure
    plt.show()

# 用于显示一组图片
def show_images_array(dataset,class_names,func=None,nrow=3,ncol=3):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):# take的是1个batch
        for i in range(nrow*ncol):
            ax = plt.subplot(nrow, ncol, i + 1)
            if func:
                images = func(images)
            #plt.imshow(images[i].numpy().astype("uint8"))
            plt.imshow(images[i]/255.0)
            plt.title(class_names[labels[i]])
            plt.axis("off")
            
def plot_multiple_images(images,nrow=3,ncol=3,figsize=(10,8),fontsize=8):
    plt.figure(figsize=figsize)
    for irow in range(nrow):
        for icol in range(ncol):
            index = irow*ncol+icol
            plt.subplot(nrow,ncol,index+1)
            plt.title(images[index]["title"],fontsize=fontsize)
            img = images[index]["image"]
            if isinstance(img,str):
                img = mpimg.imread(img)
            plt.imshow(img)
    # plt.axis('off')
    plt.show()
    
###########################################################################################
#
#               Deep Learning Processing result Display
#
#
###########################################################################################
# 图形化预测结果
#预测结束之后，可以将其绘制成图表，看看模型对于全部类的预测。
def plot_image(
    i,                 # 图片索引
    predictions_array, # 预测结果 - 多分类列表的原因是针对每个类都有一个预测值,二分类则是一个值
    true_labels,       # 实际结果
    imgs,              # 图片列表
    label_classes,      # 标签列表
    is_binary_classification = False, # 二分类还是多分类
    p=0.5
):
    true_label, img = true_labels[i], imgs[i]
    
    if is_binary_classification:
        try:
            predicted_prob = float(predictions_array)
        except:
            print(predictions_array)
        if predictions_array > p:
            predicted_label = 1
        else:
            predicted_label = 0
    else:
        predicted_label = np.argmax(predictions_array)
        predicted_prob = np.max(predictions_array) * 100
        
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("P-{} {:2.2f}% (L-{})".format(label_classes[predicted_label],
                                predicted_prob,
                                label_classes[true_label]),
                                color=color)
      
def plot_value_array(
    i,                 # 图片索引
    predictions_array,      # 预测结果 - 多分类列表的原因是针对每个类都有一个预测值,二分类则是一个值
    true_labels,          # 实际结果
    label_classes,      # 标签列表
    is_binary_classification = False,
    p=0.5
):
    true_label = true_labels[i]
    
    if is_binary_classification:
        if predictions_array>p:
            predicted_label = 1
        else:
            predicted_label = 0
        predicted_prob = float(predictions_array)
        predictions_array = [1-predicted_prob,predicted_prob]
    else:
        predicted_label = np.argmax(predictions_array)
        
    num_classes = len(label_classes)
    
    plt.grid(False)
    plt.xticks(range(num_classes))
    plt.yticks([])
    thisplot = plt.bar(range(num_classes), predictions_array, color="#777777")
    plt.ylim([0, 1])

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

#显示训练结果 
def plot_metrics(history, metrics=['precision', 'recall', 'accuracy', 'loss','auc']):
    fig, ax = plt.subplots(1, 5, figsize=(20, 3))
    ax = ax.ravel()

    for i, metric in enumerate(metrics):
        ax[i].plot(history.history[metric])
        ax[i].plot(history.history['val_' + metric])
        ax[i].set_title('Model {}'.format(metric))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(metric)
        ax[i].legend(['train', 'val'])
        
def plot_metrics_alt1(history, metrics =['precision', 'recall', 'accuracy', 'loss','auc']):
    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']      
    
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(3,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        #if metric == 'loss':
        #  plt.ylim([0, plt.ylim()[1]])
        #elif metric == 'auc':
        #  plt.ylim([0.8,1])
        #else:
        #  plt.ylim([0,1])

        plt.legend()

        
#显示训练的accuracy和loss         
def plot_model_metrics_evaluation(history,accuracy_metric_name="accuracy",loss_metric_name="loss"):
    # Retrieve a list of accuracy results on training and validation data sets for each training epoch
    acc = history.history[accuracy_metric_name]
    val_acc = history.history['val_'+accuracy_metric_name]

    # Retrieve a list of list results on training and validation data sets for each training epoch
    loss = history.history[loss_metric_name]
    val_loss = history.history['val_'+loss_metric_name]
    
    # Get number of epochs
    #epochs = range(len(acc))
    # matplotlib.pyplot.plot(*args, scalex=True, scaley=True, data=None, **kwargs)
    # x, y:这些参数是数据点的水平和垂直坐标。X为可选值。
    # 所以，这儿的epochs可省略

    # Plot training and validation accuracy per epoch
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    #plt.plot(epochs, acc, label='Training Accuracy')
    #plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    #plt.plot(epochs, loss, label='Training Loss')
    #plt.plot(epochs, loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')    
    plt.xlabel('epoch') 
    
    plt.show()
    
    return acc,val_acc,loss,val_loss

# 二次训练的accuracy和loss    
def plot_model_finetune_metrics_evaluation(history_fine,initial_epochs,acc,val_acc,loss,val_loss,accuracy_metric_name="accuracy",loss_metric_name="loss"):
    acc += history_fine.history[accuracy_metric_name]
    val_acc += history_fine.history['val_'+accuracy_metric_name]

    loss += history_fine.history[loss_metric_name]
    val_loss += history_fine.history['val_'+loss_metric_name]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.ylim([0.8, 1])
    plt.plot([initial_epochs-1,initial_epochs-1],plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.ylim([0, 1.0])
    plt.plot([initial_epochs-1,initial_epochs-1],plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    
    plt.show()
    
    return acc,val_acc,loss,val_loss


        
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 混淆矩阵
def PlotConfusionMatrix(
        y_test, # test data
        pred,  # predict 
        y_test_class0,
        y_test_class1,
        label, 
        p=0.5,
        is_save_fig=False):

    cfn_matrix = confusion_matrix(y_test,pred)
    print(cfn_matrix)
    cfn_norm_matrix = np.array([[1.0 / y_test_class0,1.0/y_test_class0],[1.0/y_test_class1,1.0/y_test_class1]])
    norm_cfn_matrix = cfn_matrix * cfn_norm_matrix

    #colsum=cfn_matrix.sum(axis=0)
    #norm_cfn_matrix = cfn_matrix / np.vstack((colsum, colsum)).T

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cfn_matrix,cmap='magma',linewidths=0.5,annot=True,ax=ax)
    #tick_marks = np.arange(len(y_test))
    #plt.xticks(tick_marks, np.unique(y_test), rotation=45)
    plt.title('Confusion Matrix',color='b')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    # if is_save_fig: plt.savefig(output_folder + '/cm_' + label + '.png')
        
    ax = fig.add_subplot(1,2,2)
    sns.heatmap(norm_cfn_matrix,cmap=plt.cm.Blues,linewidths=0.5,annot=True,ax=ax)

    plt.title('Normalized Confusion Matrix',color='b')
    plt.ylabel('Real Classes')
    plt.xlabel('Predicted Classes')
    # if is_save_fig: plt.savefig(output_folder + '/cm_norm' + label + '.png')
    plt.show()
    
    print('---Classification Report---')
    print(classification_report(y_test,pred))
    

###########################################################################################
#
#               Image Data process for machine learning
#
#
###########################################################################################

# 将图片文件转成机器学习网络处理的array
def img_to_array_batch(img_path, target_size=(150,150)):
    #import tensorflow as tf
    
    img = tf.keras.utils.load_img(
        img_path, 
        target_size=target_size, 
        interpolation="bilinear") 

    # 图片转换为Numpy array
    img_array = tf.keras.utils.img_to_array(img)
    
    # 形状转化
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    # img_array = np.array([img_array])  # 方法2
    # img_array = img_array.reshape((1,) + img_array.shape) # 方法3
    print("image shape:",img_array.shape)
    
    return img_array
    
def img_to_array_batch_alt1(img_path, target_size=(150,150)):
    # import tensorflow as tf
    
    # 加载图片
    img = tf.keras.preprocessing.image.load_img(
        img_path, 
        target_size=target_size, 
        grayscale=False, 
        color_mode="rgb", 
        interpolation="bilinear")  # this is a PIL 
    
    #如果上面没指定target_size，可以用下面方法进行resize
    #img = tf.image.resize(np.array(img), target_size)
    #img = image.resize(target_size) # 方法2
    
    # 图片转换为Numpy array
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Numpy array with shape (150, 150, 3)
    
    # 形状转化
    img_array = img_array.reshape((1,) + img_array.shape)  # Numpy array with shape (1, 150, 150, 3)
    print("image shape:",img_array.shape)
    return img_array

def image_folder_to_dataset(image_folder_path, image_size = (64, 64)):
    # import os, cv2
    import numpy as np

    # Initialize the dataset
    dataset = []

    # Loop through the images in the folder
    for filename in os.listdir(image_folder_path):
        # Load the image
        img_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(img_path)

        # Resize the image
        img = cv2.resize(img, image_size)

        # Convert the image to an array and add it to the dataset
        img_array = np.array(img)
        dataset.append(img_array)

    # Convert the dataset to a numpy array
    dataset = np.array(dataset)

    # Print the shape of the dataset
    # print(dataset.shape)

    return dataset
