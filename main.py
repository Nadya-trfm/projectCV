import os
import cv2
import numpy as np
import shutil

class_names = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
               'boat': 8,
               'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14,
               'cat': 15,
               'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23,
               'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30,
               'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35,
               'skateboard': 36,
               'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43,
               'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50,
               'carrot': 51,
               'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58,
               'bed': 59,
               'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66,
               'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73,
               'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}

file_name = input('Введите путь до файла: ')
class_name = input('Введите название объекта, который хотите найти: ')
while class_name not in class_names:
    class_name = input('Этот объект нельзя найти, введите другой: ')
# берем параметры исходного видео
cap = cv2.VideoCapture(file_name)
if cap.isOpened() == False:
    print("Ошибка открытия видеофайла")
else:
    # находим колличество кадров в видео, фпс, ширину и длину
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(5))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# делаем запрос в detect.py
zapros = 'python yolov5/detect.py --source ' + file_name +' --save-txt --classes ' + str(class_names[class_name]) \
         + ' --project result --name work --save-crop --hide-labels'
os.system(zapros)

# создаём черные кадры размеров видео
for i in range(1, number_of_frames + 1):
    img = np.zeros((video_height, video_width, 1), dtype="uint8")
    fragment = 'result/work/is_{0:04d}.jpg'.format(i)
    cv2.imwrite(fragment, img)

file_count = 1  # счетчик вырезанных фрагментов
black_file = 1  # счетчик кадров в видео
a = file_name.rfind('/')
video_name = file_name[a+1:-4]  # обрезаем название видео

# соединяем найденные фрагменты с черным фоном
for i in range(number_of_frames):
    # берем черную картинку соответсвующего кадра
    black_img = 'result/work/is_{0:04d}.jpg'.format(black_file)
    l_img = cv2.imread(black_img)

    file_name = 'result/work/labels/{0}_{1}.txt'.format(video_name, black_file)
    # проверяем существование файла с параметрами фрагментов, если такого нет - сохраняем ту же черную картинку
    if os.path.isfile(file_name) == False:
        fragment = 'result/work/is_{0:04d}.jpg'.format(black_file)
        cv2.imwrite(fragment, l_img)
        black_file += 1
        continue
    f = open(file_name, 'r')
    line_list = f.readlines()
    f.close()

    for line in line_list:
        # считываем параметры, cl-класс найденного объекта, res1,res2-x,y центра фрагмента; res3,res4- ширина и длина
        # фрагмента
        cl, res1, res2, res3, res4 = [float(i) for i in line.split()]
        # считываем фрагмент соответсвующей строки
        dob_file = '' if file_count == 1 else str(file_count)
        fragment = 'result/work/crops/' + class_name + '/' + video_name + dob_file + '.jpg'
        s_img = cv2.imread(fragment)
        # находим координаты верхнего левого угла фрагмента для его вставки
        xCenter = video_width * res1
        yCenter = video_height * res2
        rectWidth = video_width * res3
        rectHeight = video_height * res4
        x_offset = int(xCenter - 0.5 * rectWidth)
        y_offset = int(yCenter - 0.5 * rectHeight)
        width = int(s_img.shape[1])
        height = int(s_img.shape[0])
        # если координаты фрагмента больше координат картинки-обрезаем фрагмент
        if x_offset + width > video_width:
            s_img = s_img[0:height, 0:video_width - (x_offset + width)]
        if y_offset + height > video_height:
            s_img = s_img[0:video_height - (y_offset + height), 0:width]
        file_count += 1
        # вставляем фрагмент в черную картинку
        l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    # перезаписываем кадр
    fragment = 'result/work/is_{0:04d}.jpg'.format(black_file)
    cv2.imwrite(fragment, l_img)
    black_file += 1
vid_capture = cv2.VideoCapture('result/work/is_%04d.jpg')
frame_size = (video_width, video_height)
new_name = 'output'+video_name+'.avi'
# запись готового видео
output = cv2.VideoWriter(new_name, cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)
while(vid_capture.isOpened()):
    ret, frame = vid_capture.read()
    if ret == True:
           output.write(frame)
    else:
        break
vid_capture.release()
output.release()
# удаляем мусор
path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result/work')
shutil.rmtree(path)