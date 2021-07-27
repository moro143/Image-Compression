import numpy as np
import cv2
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, numWeights):
        self.weights = np.random.uniform(-0.5, 0.5, numWeights+1)
    
    def get_value(self, input):
        values = input*self.weights
        value = np.sum(values)
        return 1 / (1+np.e**(-value))
    
    def update_weights(self, de, n):
        self.weights = self.weights - ((1/(n**(1/4)))*de)

def error(d, y):
    return 1/2*np.sum((d-y)**2)

def img_to_01(src, t = True):
    img = cv2.imread(src)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    listgray = []
    for i in gray:
        for j in i:
            listgray.append(j/255)
    return np.array(listgray)

def to_chuncks(img, x=100, y=100):
    list_img_chunks = []
    for i in range(int(x/10)):
        for j in range(int(y/10)):
            list_img_chunks.append(img[10*i:10*i+10, 10*j:10*j+10])
    return list_img_chunks

def from_chuncks(list_img_chunks, x=100, y=100):
    img = np.ones((x, y))
    c = 0
    for i in range(int(x/10)):
        for j in range(int(y/10)):
            img[10*i:10*i+10, 10*j:10*j+10] = list_img_chunks[c]
            c+=1
    return img

def forward(layers, X):
    values = X
    values = np.append(values, 1.0)
    all_values = [values]
    for l in layers:
        tvalues = []
        for v in l:
            neuron_output = v.get_value(values)
            tvalues = np.append(tvalues, neuron_output)
        values = np.append(tvalues, 1.0)
        all_values.append(values)
    return all_values



def step(layers, x,d, k, name, train=True):
    all_values = forward(layers, x)
    y = all_values[-1][:-1]
    if train:
        de = []
        for v in all_values[-2]:
            de.append(-(d-y)*y*(1-y)*v)
        de = np.transpose(de)
        for n in range(len(layers[-1])):
            layers[-1][n].update_weights(de[n], k)
        
        x = -(d-y)*y*(1-y)
        ss = []
        for w in range(len(layers[-1][0].weights)):
            s = 0
            for n in range(len(layers[-1])):
                s+=layers[-1][n].weights[w]*x[n]
            ss.append(s)
        sums = ss
        de = []
        for i in all_values[0]:
            de.append(sums*all_values[-2]*(1-all_values[-2])*i)
        de = np.transpose(de)[:-1]
        for n in range(len(layers[-2])):
            layers[-2][n].update_weights(de[n], k)
    return error(d,y)

# Zbior do nauki
img = cv2.imread('train.tiff')
x, y, _ = np.shape(img)
imge = img_to_01('train.tiff')
imge = np.reshape(imge, (x,y))
lista = to_chuncks(imge, x, y)
t = []
for i in lista:
    t.append(np.reshape(i, (100)))
trains = t
np.random.shuffle(trains)
test = t
def create(num_in):
    layer1 = []
    for _ in range(num_in):
        layer1.append(Neuron(100))
    layer2 = []
    for _ in range(100):
        layer2.append(Neuron(num_in))
    layers = [layer1, layer2]
    return layers

def train(layers, dk=1, times=10):
    k = dk
    errs = []
    try:
        for _ in range(times):
            tmp = 0
            for i in trains:
                er = step(layers, i,i,  k, 't.png')
                k+=dk
                errs.append(er)
                tmp += er
            print(tmp/len(trains))
            if tmp/len(trains)<=0.19:
                break
            k+=dk
    except KeyboardInterrupt:
        pass
    return errs, layers

dks = [1]

for dk in dks:
    layers = create(10)
    s, layers = train(layers, dk, 1000)
    plt.plot(s)

plt.legend(dks)
plt.show()

img = cv2.imread('train.tiff')
print(np.shape(img))
x, y, _ = np.shape(img)
imge = img_to_01('train.tiff')
imge = np.reshape(imge, (x,y))
lista = to_chuncks(imge,x,y)
# zapisanie wartosci zdjecia
done = []
for i in lista:
    tmp = np.reshape(i,100)
    done.append((forward(layers,tmp)[1]))
np.save('values.npy', done)
# zapisanie wag 
weights = []
for n in layers[-1]:
    weights.append(n.weights)
np.save('weights.npy', weights)

img = cv2.imread('train.tiff')
x, y, _ = np.shape(img)
imge = img_to_01('train.tiff')
imge = np.reshape(imge, (x,y))
lista = to_chuncks(imge,x,y)
done = []
c=0
for i in lista:
    tmp = np.reshape(i,100)
    done.append(np.reshape(forward(layers,tmp)[-1][:-1], (10,10)))
    c += 1
np.save('shape.npy', [x,y]) # zapisanie shape zdjecia

imgdone = from_chuncks(done, x, y)
cv2.imshow('k', imgdone)
cv2.imwrite('decompressed_image.png', 255*imgdone)
cv2.waitKey()