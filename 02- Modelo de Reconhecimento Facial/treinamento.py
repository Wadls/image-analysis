import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('fotos',f) for f in os.listdir('fotos')] #Determinando o caminho na pasta fotos
    #print(caminhos) esse código é também um teste, que mostra o nome das imagens que ele abre
    faces = []
    ids = []
    for caminhoImagem in caminhos: #Aqui ele percorre todas as imagens dentro da nossa pasta fotos
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow('Face',imagemFace) Caso queira ver na prática oq ele vai, é só liberar
        #cv2.waitKey(30) essas duas linhas de código
    return np.array(ids), faces

ids, faces = getImagemComId()
#print(ids) Código de Observação, mostra quantos ids ele reconheceu no total

print('Treinando...')
eigenface.train(faces,ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces,ids)
fisherface.write("classificadorFisher.yml")

lbph.train(faces,ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento realizado')