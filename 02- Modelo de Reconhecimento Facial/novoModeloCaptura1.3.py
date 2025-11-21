import cv2
import numpy as np
#É a mesma versão da 1.3, Porém o objetivo desse sistema é melhorar a sua capacidade de reconhecer faces, usando o haarcascade eyes

classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('haarcascade_eye.xml')
# 1. Na Variável Vídeo é atribuída o vídeo a uma variavel, pode ser o vídeo de uma webcam tbm
video = cv2.VideoCapture('Video02.0.mp4')
amostra = 1 #Numero da primeira amostra, que vai ser incrementada no programa
numeroAmostra = 25 #Valor Mínimo de amostras a serem coletadas para treinar o modelo
id = input('Digite O identificador da Pessoa: ')
largura,altura = 500,500

# Verifica se o vídeo abriu corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo! Verifique o caminho ou o arquivo.")
else:
    # 2. Esse loop 'while' serve para ler cada frame do vídeo
    while True:
        # 3. 'read()' retorna um booleano (leitura) e o frame (imagem)
        leitura, frame = video.read() #O Video.Read do cv2 Retorna Falso se o vídeo não for encontrado/lido
        
        # 4. Se 'leitura' for False, o vídeo terminou ou deu erro
        if not leitura:
            print("Vídeo terminou ou ocorreu um erro na leitura.")
            break

        # 5. Agora sim, fazemos o processamento no 'frame' (na imagem atual)
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                         #scaleFac+tor=1.5, Funciona pra WebCam, mas não posso usar em um vídeo, pois vai ficar super acelerado
                                                         minSize=(100,100))

        # 6. Desenha os retângulos no 'frame' original (colorido)
        for (x, y, l, a) in facesDetectadas:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
            regiao = frame[y:y + a,x:x + l]
            regiaoCinzaOlho= cv2.cvtColor(regiao,cv2.COLOR_BGR2GRAY)
            olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlho)
            
            for (ox,oy,ol,oa) in olhosDetectados:
                cv2.rectangle(regiao,(ox,oy),(ox + ol, oy + oa), (0,255,0), 2)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    #if np.average(imagemCinza) > 110:
                        imagemFace = cv2.resize(imagemCinza[y:y+a,x:x+l],(largura,altura))
                        cv2.imwrite('fotos/pessoa.'+str(id)+'.'+str(amostra)+'.jpg',imagemFace)
                        print('[foto'+str(amostra)+'capturada com sucesso]')
                        amostra+=1
                
            

        cv2.imshow('Detecção Facial', frame)
        cv2.waitKey(1)
        if amostra>= numeroAmostra+1:
            break
        

# 9. Libera os recursos (importante!)
video.release()
cv2.destroyAllWindows()