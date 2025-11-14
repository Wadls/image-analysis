import cv2

algoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') #Essa função serve para carregar o algoritmo padrão de reconhecimento de rostos da biblioteca open cv

imagem = cv2.imread('BD/Imagem-7.png') #Atribuindo alguma imagem presente no meu banco

imagemCinza = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY) #O sistema reconhece rostos com maior facilidade se as imagens estiverem em escala de cinza

faces = algoritmo.detectMultiScale(imagemCinza) #Logo o algoritmo detecta os rostos sobre a imagem em escala de cinza

print(faces) #Essas são as coordenadas que foram geradas pelo algoritmo, em ordem elas representam o eixo x e y, largura e altura do rosto

for (x,y,l,a) in faces: # x = Eixo x, Y= Eixo y, L= Largura do retângulo, A = Altura do Retângulo
    cv2.rectangle(imagem, (x,y),( x + l , y + a ),(0,255,0),1) #Os dois primeiros argumento servem para dar o endereço do retângulo, o 3 argumento define a cor do retângulo em RGB, e o quarto argumento define o grossura do retângulo

cv2.imshow('Minha Imagem',imagem) #Os Desenhos são feitos por cima da variável imagem, logo a partir daqui a imagem desse endereço de memória do sistema sempre estará sobreposta com o retângulo

cv2.waitKey(0) # Espera você apertar qualquer tecla
cv2.destroyAllWindows() # Fecha a janela
