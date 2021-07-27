import pickle
import numpy as np
import cv2

compressed_img = np.load('values.npy')
weights = np.load('weights.npy')
shape = np.load('shape.npy')

def from_chuncks(list_img_chunks, x=100, y=100):
    img = np.ones((x, y))
    c = 0
    for i in range(int(x/10)):
        for j in range(int(y/10)):
            img[10*i:10*i+10, 10*j:10*j+10] = list_img_chunks[c]
            c+=1
    return img
chunks = []
for partImg in compressed_img:
    chunk = []
    for w in weights:
        value = (w@partImg)
        chunk.append((1 / (1+np.e**(-value))))
    chunks.append(np.reshape(chunk,(10,10)))
done = from_chuncks(chunks, shape[0], shape[1])
cv2.imshow('k', done)
cv2.waitKey()