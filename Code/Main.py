import siamese as Model
import utils as utils
import numpy as np
model = Model()
util = utils()
imageA,imageB,label = util.loadData('/home/alifathi/Documents/AI/Git/siameseNetwork/Data/verification_dev.csv')

model.compileModel()
model.train(np.concatenate([imageA,imageB]),np.array(label))