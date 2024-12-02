

if not os.path.exists("/binhe/models/yolov3.weights"):
    !wget https://pjreddie.com/media/files/yolov3.weights -O /binhe/models/yolov3.weights
        
import os
import cv2
import numpy as np

weightsPath="/binhe/models/yolov3.weights"
configPath="/binhe/repos/darknet_AlexeyAB_tube_135/cfg/yolov3.cfg"
labelsPath="/binhe/repos/darknet_AlexeyAB_tube_135/cfg/coco.names"
#图像读取地址
rootdir = "/binhe/dataset/ShenDiaoXiaLve/Index"
# 图像保存地址
savepath = "/binhe/dataset/ShenDiaoXiaLve/Index-Cropped"  

def extract_object(
            weightsPath,
            configPath,
            labelsPath,
            rootdir,
            savepath):
    #初始化一些参数
    LABELS = open(labelsPath).read().strip().split("\n")  #物体类别
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")#颜色
    print(LABELS)
    # print(COLORS)

    filelist = os.listdir(rootdir)  # 打开对应的文件夹
    total_num = len(filelist)  #得到文件夹中图像的个数
    print("totle files: ", total_num)
    # 如果输出的文件夹不存在，创建即可
    if not os.path.isdir(savepath):
        os.makedirs(savepath)

    for(dirpath,dirnames,filenames) in os.walk(rootdir):
        for filename in filenames:
            # 必须将boxes在遍历新的图片后初始化
            boxes = []
            confidences = []
            classIDs = []
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            path = os.path.join(dirpath,filename)
            image = cv2.imread(path)
            (H, W) = image.shape[:2]
            # 得到 YOLO需要的输出层
            ln = net.getLayerNames()        
            try:
                ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            except:
                ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]    
            #从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            #在每层输出上循环
            for output in layerOutputs:
                # 对每个检测进行循环
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    #过滤掉那些置信度较小的检测结果
                    if confidence > 0.5:
                        #框后接框的宽度和高度
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        #边框的左上角
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        # 更新检测出来的框
                       # 批量检测图片注意此处的boxes在每一次遍历的时候要初始化，否则检测出来的图像框会叠加
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            # 极大值抑制
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
            k = -1
            if len(idxs) > 0:
                # for k in range(0,len(boxes)):
                for i in idxs.flatten() :
                    #print(LABELS[classIDs[i]])
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # 在原图上绘制边框和类别
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    # image是原图，     左上点坐标， 右下点坐标， 颜色， 画线的宽度
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    # 各参数依次是：图片，添加的文字，左上角坐标(整数)，字体，        字体大小，颜色，字体粗细
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
                    # 图像裁剪注意坐标要一一对应
                    # 图片裁剪 裁剪区域【Ly:Ry,Lx:Rx】
                    cut = image[y:(y+h), x:(x + w)]
                    # boxes的长度即为识别出来的车辆个数，利用boxes的长度来定义裁剪后车辆的路径名称
                    if k < len(boxes):
                        k = k+1
                    # 从字母a开始每次+1
                    t = chr(ord("a")+k)
                    # 写入文件夹，这块写入的时候不支持int（我也不知道为啥），所以才用的字母
                    outputfilename = filename.split(".")[0] + "_" + t + "_" + LABELS[classIDs[i]] + ".jpg"
                    outputfilepath = os.path.join(savepath, outputfilename)
                    print(outputfilepath)
                    cv2.imwrite(outputfilepath,cut)    