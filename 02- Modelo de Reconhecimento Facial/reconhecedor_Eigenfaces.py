import cv2
import numpy as np

classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('haarcascade_eye.xml')
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read('classificadorEigen.yml')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
video = cv2.VideoCapture('Video02.mp4')
largura,altura = 500,500

if not video.isOpened():
    print("Erro ao abrir o vídeo! Verifique o caminho ou o arquivo.")
else:
    while True:
        leitura, frame = video.read()
        
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                         minSize=(50,50))

        for (x, y, l, a) in facesDetectadas:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
            regiao = frame[y:y + a,x:x + l]
            regiaoCinzaOlho= cv2.cvtColor(regiao,cv2.COLOR_BGR2GRAY)
            olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho)
            
            for (ox,oy,ol,oa) in olhosDetectados:
                imagemFace = cv2.resize(imagemCinza[y:y+a,x:x+l],(largura,altura))
                cv2.rectangle(regiao,(ox,oy),(ox + ol, oy + oa), (0,255,0), 2)
                id, confianca = reconhecedor.predict(imagemFace)
                if id == 1:
                    nome = 'Mount'
                elif id == 2:
                    nome = 'Calvo Aleatorio'
                else:
                    nome = "Unknow"
                
                cv2.putText(frame,nome,(x+50,y+(a+30)), font,2,(0,0,255))
                    
                cv2.putText(frame,str(id) + ':',(x,y+(a+30)), font,2,(0,0,255)) #Essa Versão do Código usa o id no lugar da variável nome
                cv2.putText(frame,str(confianca),(x,y+(a+50)), font,1,(0,0,255)) #Mostra a porcentagem de confiabilidade junto com o nome
                    
                    
            
        cv2.imshow('Face', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
        
video.release()
cv2.destroyAllWindows()