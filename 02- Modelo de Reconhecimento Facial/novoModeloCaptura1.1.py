import cv2

classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 1. Renomeei a variável para 'video' para ficar mais claro
video = cv2.VideoCapture('Video01.mp4')

# Verifica se o vídeo abriu corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo! Verifique o caminho ou o arquivo.")
else:
    # 2. Cria um loop 'while' para ler cada frame
    while True:
        # 3. 'read()' retorna um booleano (ret) e o frame (imagem)
        leitura, frame = video.read() #O Video.Read do cv2 Retorna Falso se o vídeo não for encontrado/lido

        # 4. Se 'leirua' for False, o vídeo terminou ou deu erro
        if not leitura:
            print("Vídeo terminou ou ocorreu um erro na leitura.")
            break

        # 5. Agora sim, fazemos o processamento no 'frame' (na imagem atual)
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                         #scaleFactor=1.5, Funciona pra WebCam, mas não posso usar em um vídeo, pois vai ficar super acelerado
                                                         minSize=(100,100))

        # 6. Desenha os retângulos no 'frame' original (colorido)
        for (x, y, l, a) in facesDetectadas:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)

        # 7. Mostra o 'frame' processado na janela
        cv2.imshow('Detecção Facial', frame)

        # 8. Espera 1ms e verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 9. Libera os recursos (importante!)
video.release()
cv2.destroyAllWindows()