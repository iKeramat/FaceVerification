import pandas as pd
from sklearn.model_selection import train_test_split
import cv2 as cv
class utils():
    def loadData(self,manAddr):
        man = pd.read_csv(manAddr)
        imagesA = man['image_A']
        imagesB = man['image_B']
        label = man['match']
        imagesA = imagesA.apply(lambda addr: self.loadAndPrerocess('../Data/'+addr))
        imagesB = imagesB.apply(lambda addr: self.loadAndPrerocess('../Data/'+addr))
        return [imagesA,imagesB,label]

    @staticmethod
    def loadAndPrerocess(path):
        image = cv.imread(path)
        image = cv.resize(image,(64,64))
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image/255.0
        return image

    @staticmethod
    def Split(images,label):
        return train_test_split(images,label,test_size=0.2)


    def visualizeData(self):
        pass
