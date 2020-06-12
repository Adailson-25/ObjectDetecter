import imutils                                                                         #processamento de imagem comumente usadas como redimensionar, girar, traduzir etc.
import numpy as np                                                                     #pacote para a linguagem Python que suporta arrays e matrizes multidimensionais.

def meanSquareError(img1, img2):                                                       # Calcula o erro quadrático médio entre duas matrizes n-d. Inferior = mais semelhante.
    assert img1.shape == img2.shape, "Images must be the same shape."
    error = np.sum((img1.astype("float") - img2.astype("float")) ** 2)                 #105638752.0
    error = error/float(img1.shape[0] * img1.shape[1] * img1.shape[2])                 #13022.528599605523
    return error

def compareImages(img1, img2):
    return 1/meanSquareError(img1, img2)

def pyramid(image, scale = 1.5, minSize = 30, maxSize = 1000):                         # Calcula pirâmides de imagens (começa com as amostras originais e inferiores).
    #yield image
    while True:
        w = int(image.shape[1] / scale)                                                #salva um numero inteiro pela divisão de tamanho da imagem pela escala
        image = imutils.resize(image, width = w)                                       #redimensionar a imagem conforme o novo tamanho da imagem
        if(image.shape[0] < minSize or image.shape[1] < minSize):                      #para a função se a imagem estiver no menor tamanho = 30
            break
        if (image.shape[0] > maxSize or image.shape[1] > maxSize):                     #continua a função
            continue
        yield image

def sliding_window(image, stepSize, windowSize):                                       # "Desliza" uma janela sobre a imagem.
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[1]])

import argparse
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the target image")         
ap.add_argument("-p", "--prototype", required=True, help="Path to the prototype object")
args = vars(ap.parse_args())

targetImage = cv2.imread(args["image"])                                                  #salva a imagem em targetImage                                               
targetImage = imutils.resize(targetImage, width=500)                                     #redimensiona a imagem salva em targeImage
prototypeImg = cv2.imread(args["prototype"])                                             #salva a imagem a ser procurada em prototyImg
maxSim = -1
maxBox = (0,0,0,0)

t0 = time.time()                                                                          #inicio do tempo de execução
for p in pyramid(prototypeImg, minSize = 50, maxSize = targetImage.shape[0]):             #chama a função que gera a piramide salvando a imagem do prototipo em p
    #i = 1
    for (x, y, window) in sliding_window(targetImage, stepSize = 2, windowSize = p.shape):#chama a função que cria uma imagem de cada parte da imagem principal
        #cv2.imwrite("imagem{}.jpg".format(i), window)
        #i = i+1
        if window.shape[0] != p.shape[0] or window.shape[1] != p.shape[1]:                #compara se os parametros da imagem são iguais
             continue

        tempSim = compareImages(p, window)                                                #commpara as imagens p(prototipo) e window(principal)
        if(tempSim > maxSim):                                                             
            maxSim = tempSim
            maxBox = (x, y, p.shape[0], p.shape[1])

t1 = time.time()                                                                          #fim do tempo de execução

print("Execution time: " + str(t1 - t0))
print(tempSim)
print(maxSim)
print(maxBox)
buff1 = 10
(x, y, w, h) = maxBox

cv2.rectangle(targetImage,(int(x-buff1),int(y-buff1/2)),(int(x+w),int(y+h)),(0,255,0),2)


cv2.imshow('image', targetImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
