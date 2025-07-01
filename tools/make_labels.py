"""
Программа для разметки изображений
на вход необходимо передать 2 параметра:
 --list  |  список изображений
 --class |  номер класса который будете размечать
 Рядом с изображением появится файл с расширением .txt в котором будет разметка в формате YOLO Darknet

Программа отображает уже существующую раметку для выбранного класса. 
Разметка для других классов не отображается, т.е. программа позволяет работать только с одним классом объектов в один момент времени.
Если надо разметить более чем один класс, то запустите программу повторно и нужным номером класса, столько раз сколько классов необходимо разметить.
Разметка будет создаваться в директории с изображениями.

Пример запуска:
python3 marking.py --list list.csv --class 0

Клавиши:
a - предыдущее изображение
d - следующее изображение
Del - удалить выделенный контур. Контур выделяется кликом мыши внути контура 
Backspace -  удаление изоражения из списка 
Esc - выход
"""
import cv2
import numpy as np
import copy
import pandas as pd
import argparse
import os



indexPics = 0 # index current image
classId = 0
tempColor = (0,255,0) # color of polygon that haven't finished drawing
finishColor = (0,255,255) # color finished contours
thickness = 3 # thickness of line contours
thDistance = 1 # min distance from click point to contour

"""
This function calculates the square of the vector length

Args:
    v: vector. Type numpy.ndarray.

Returns:
    Square of the vector length. Type numpy.float64
"""
def magnitude(v):
    return np.sum(np.fromiter((vi ** 2 for vi in v), float))


"""
This function find minimal distance between point and segment in 2-D

Args:
    a: Coordinates of point A of the segment AB. Type numpy.ndarray.
    b: Coordinates of point B of the segment AB. Type numpy.ndarray.
    p: Coordinates of point, distance for that find.

Returns:
    Minimal distance between point and segment. Type numpy.float64
"""
def minDistance(a, b, p): # a, b - coords of line, p - coords of click point
    ab = b-a
    bp = b-p
    pa = p-a
    dAB = magnitude(ab)
    dBP = magnitude(bp)
    dPA = magnitude(pa)
    if dPA > (dBP + dAB) or dBP > (dPA + dAB):
        if dPA < dBP:
            minDistance = np.sqrt(dPA)
        else:
            minDistance = np.sqrt(dBP)
    else:
        minDistance = abs(ab[0]*p[1] - ab[1]*p[0] + b[1]*a[0] - b[0]*a[1])/np.sqrt(dAB)
    return minDistance

def distanceToBox(box, point):
    x_min = min(box[0], box[2])
    x_max = max(box[0], box[2])
    y_min = min(box[1], box[3])
    y_max = max(box[1], box[3])    
    if x_min <= point[0] and point[0] <= x_max and y_min <= point[1] and point[1] <= y_max:
        return abs((box[2] - box[0]) * (box[3] - box[1]))
    else: 
        return 1.

"""
This function generates list of minimal distances from point to each contours

Args:
    array: list of contours
    point: coordinates of point

Returns:
    List of minimal distances from point to each contours.
"""
def distanceToContours(array, point):
    distToContours = [] # list of minimal distances from point to each contour
    for box in array:
        distToContours.append(distanceToBox(box, point)) # choose minimal distance to contour
    return distToContours


"""
This function read json-file with structure:
# json file example
[
    {
            "contour" : 
                [
                    {
                        "x" : 0,
                        "y" : 0
                    },
                    {
                        "x" : 10,
                        "y" : 10
                    }
                ]
    },
    {
    }
]

and get list of contours.

Args:
    filename: name of json-file 

Returns:
    List of contours. Element of list is contour. Every contour have a type numpy.ndarray, dimention (n, 2), n-number points in contour.  
"""
def loadLabels(filename):
    arrayContour = []
    labels = pd.DataFrame()
    
    try:
        labels = pd.read_csv(filename, header=None, sep=' ')
    except Exception:
        pass
    
    for element in labels.to_numpy():
            pts = np.array ([])
            if element[0].astype('int32') != classId:
                continue
            element = element.astype('float')
            x = element[1]
            y = element[2]
            w = element[3]
            h = element[4]
            x = x - w/2 #YOLO type structure rectangle
            y = y - h/2
            currentContour = np.array([])
            currentContour = np.append(currentContour, [x,y])
            currentContour = np.append(currentContour, [x+w,y+h])
            arrayContour.append(currentContour)        

            


    return arrayContour

"""
This function save contour in text file as json

Args:
    filename: name of json-file
    arrayContour: array of contours
"""
def saveLabels(filename, arrayContour):
    labels = pd.DataFrame({0:[],1:[],2:[],3:[],4:[]})
    try:
        labels = pd.read_csv(filename, header=None, sep=' ')
    except FileNotFoundError:
        pass
    except Exception:
        print('Can`t read file labels ', filename)
        
    #оставляем только те метки которые не редактируем
    labels = labels[labels[0] != classId]
    
    #добавляем метки которые редактировали
    for contour in arrayContour:
        x = (contour[0] + contour[2])/2
        y = (contour[1] + contour[3])/2
        w = abs(contour[0] - contour[2])
        h = abs(contour[1] - contour[3])
        df = pd.DataFrame([[classId, x, y, w, h]], columns=[0,1,2,3,4])
        labels = pd.concat([labels, df], ignore_index=True)

    labels[0] = labels[0].astype('int')
    labels.to_csv(filename, header=False, index=False, sep=' ')

"""
This function load image by index and create new window, loading early saved contours for this image

Args:
    indexPics: index of new image

Returns:
    filename - name of json-file
    numImage - name image
    pic - deepcopy of source image with drawn contours
    arrayContour - array of contours
    img - loaded image
"""
#def createImage(indexPics): # load data and create new window
    #global drawing
    #global crossline
    #global selectedContour
    #global indexSelectedContour
    #global currentContour
    
    #drawing = False
    #crossline = False
    #selectedContour = False # flag mean that we choose contour for delete
    #indexSelectedContour = -1 # index of contour that we choose
    #currentContour = np.array ([]) # array  points of polygon that haven't finished drawing
    #numImage = pics.iloc[indexPics][0]
    #img = cv2.imread(numImage)
    #filename = numImage+'.json' 
    #arrayContour = loadJson(filename)
    #pic = drawContours(img, arrayContour, finishColor)
    #cv2.namedWindow(numImage, flags= cv2.WINDOW_GUI_NORMAL ) #| cv2.WINDOW_AUTOSIZE settings params window without dropdown menu and with image size
    #cv2.setMouseCallback(numImage,mousePosition)
    #return filename, numImage, pic, arrayContour, img
    



   
"""
This class work with contour fot selected image

Args:
    imageFileName: filename image
"""
class Contour:
    flagMakedWindow = False
    nameWindow = "main"
    
    @classmethod
    def makeWindow(self):
        if not self.flagMakedWindow:
            cv2.namedWindow(self.nameWindow, flags= cv2.WINDOW_GUI_NORMAL ) 
            #| cv2.WINDOW_AUTOSIZE settings params window without dropdown menu and with image size
            self.flagMakedWindow = True
            
    
    def __init__(self, imageFileName):
        self.imageFileName = "main"
        self.drawing = False
        #self.crossline = False
        self.selectedContour = False # flag mean that we choose contour for delete
        self.indexSelectedContour = -1 # index of contour that we choose
        self.currentContour = np.array ([]) # array  points of polygon that haven't finished drawing
        self.img = cv2.imread(imageFileName)
        title, ext = os.path.splitext(imageFileName)
        self.jsonFileName = title +'.txt'
        print(self.jsonFileName)
        self.arrayContour = loadLabels(self.jsonFileName)
        self.pic = self.drawContours(self.img, self.arrayContour, finishColor)
        
        self.makeWindow()
        cv2.setMouseCallback(self.nameWindow, self.mousePosition)
    
    """
    This method draw poligons on image

    Args:
        image: source image
        contours: array finished contours
        color: color finished contours

    Returns:
        Deep copy of source image with finished contours
    """
    def drawContours(self, img, countours, color): 
        imgCopy = copy.deepcopy(img)
        
        for curr in countours:
            c = copy.copy(curr)
            c[1::2] *= img.shape[0]
            c[0::2] *= img.shape[1]
            c = c.astype('int')
            c = c.reshape(-1,2)
            cv2.rectangle(imgCopy, tuple(c[0]), tuple(c[1]), color, thickness)

        return imgCopy
    

    def destroy(self):
        # save data and destroy window
        #cv2.destroyWindow(self.imageFileName)
        saveLabels(self.jsonFileName, self.arrayContour)
        
    def deleteContour(self):
        if self.selectedContour:
            self.arrayContour.pop(self.indexSelectedContour)
            self.selectedContour = False
            self.pic = self.drawContours(self.img, self.arrayContour, finishColor)
            
        
    # mouse callback function
    def mousePosition(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
                self.pic = self.drawContours(self.img, self.arrayContour, finishColor)
                # if the first click and array of finished contours have coordinates, we test distance between point and contours
                if self.currentContour.shape[0] == 0 and len(self.arrayContour) > 0:
                    listDist = distanceToContours(self.arrayContour, np.array([x/self.img.shape[1],y/self.img.shape[0]]))
                    md = min(listDist)
                    self.indexSelectedContour = listDist.index(md)
                    if md < thDistance: # if minimal distance from point to contour smaller threshhold => choose contour
                        self.selectedContour = True
                        #cv2.polylines (self.pic, [self.arrayContour[self.indexSelectedContour]], True , (0,0,255), thickness)
                        self.pic = self.drawContours(self.pic, [self.arrayContour[self.indexSelectedContour]], (0,0,255))
                        cv2.putText(self.pic, "press 'Del' to delete", (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        return
                    
                self.drawing = True
                # add point coords to currentContour
                self.currentContour = np.append(self.currentContour, [x/self.img.shape[1], y/self.img.shape[0]])


                if self.currentContour.shape[0] == 4:  # if polyline`s points in array >1, draw polyline
                    self.drawing = False
                    self.selectedContour = False
                    self.arrayContour.append(self.currentContour)
                    self.currentContour = np.array ([])
                    self.pic = self.drawContours(self.img, self.arrayContour, finishColor)
                    
        if event == cv2.EVENT_MOUSEMOVE:
                self.pic = copy.deepcopy(self.img)
                cv2.line(self.pic, (0,y), (self.pic.shape[1],y), (255,0,0), thickness)
                cv2.line(self.pic, (x,0), (x,self.pic.shape[0]), (255,0,0), thickness)
                self.pic = self.drawContours(self.pic, self.arrayContour, finishColor) 
                
                if self.currentContour.shape[0] > 0 and self.drawing == True:
                        currentContour2 = np.append(self.currentContour, [x/self.img.shape[1], y/self.img.shape[0]])

                        
                        self.pic = self.drawContours(self.pic, [currentContour2], tempColor)
                        
                    
        #stop draw current contours and add points to array contours
        elif event == cv2.EVENT_RBUTTONDOWN: 
            if self.selectedContour:
                self.deleteContour()
            self.drawing = False
            self.selectedContour = False
            self.pic = self.drawContours(self.img, self.arrayContour, finishColor)  
            self.currentContour = np.array ([])

    def imShow(self):
        cv2.imshow(self.imageFileName, self.pic)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--class", type=int, default=0, help="set classId")
ap.add_argument("-l", "--list", type=str, default="list.csv", help="list file images")

args = vars(ap.parse_args())
classId = args["class"]
listFile = args["list"]

pics = pd.read_csv(listFile, header=None) # load list images
pics = pics.sort_values(0)
#print(pics)
print('number images in list: ',pics.shape[0])

if len(pics)==0:
    print('List is empty')
    pass

newIndexPic = indexPics
#filename, numImage, pic, arrayContour, img = createImage(indexPics) 

contour = Contour(pics.iloc[indexPics][0]) #create the first image

flagChange = False

while(1):
    contour.imShow()
    k = cv2.waitKey(30)
    #print(k)
    if k == 27: # Esc
        break
    if k == 8: # Backspace
        print(indexPics)
        pics.drop([indexPics], axis=0, inplace=True)
        pics.reset_index(drop=True, inplace=True)
        newIndexPic = indexPics
        indexPics = -1
        flagChange = True
    if k == 255: # Del
        contour.deleteContour()
    if k == 97: # Left bottom 'a'
        newIndexPic = indexPics -1
    if k == 100: # Right bottom 'd'
        newIndexPic = indexPics + 1
    if newIndexPic >= len(pics) or newIndexPic <= -len(pics): # new cycle of list with pics
        newIndexPic = 0
    if newIndexPic != indexPics:
        contour.destroy()
        indexPics = newIndexPic
        contour = Contour(pics.iloc[indexPics][0]) #create next image
        

cv2.destroyAllWindows()
if flagChange:
    print("File list.csv is changed, save?[Y/n]")
    answ = input()
    if answ == "Y" or answ == "y" or answ == "":
        pics.to_csv(listFile, header=False, index=False)
