import cv2
import glob # Biblioteca para encontrar caminhos de arquivos

#Essa função serve para carregar o algoritmo padrão de reconhecimento de rostos da biblioteca open cv
algoritmo = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Busca todas as imagens jpg dentro da pasta BD
# *.jpg = qualquer arquivo que termine em .jpg
lista_imagens = glob.glob('BD/*.jpg') 



# Inicia o loop para processar uma por uma
for caminho_imagem in lista_imagens:
    print(f"Processando: {caminho_imagem}")
    
    imagem = cv2.imread(caminho_imagem)
    
    # Verificação de segurança caso a imagem esteja corrompida
    if imagem is None:
        print(f"Erro ao abrir {caminho_imagem}")
        continue # Pula para a próxima imagem do loop

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    faces = algoritmo.detectMultiScale(
        imagemCinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, l, a) in faces:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

    
    # Mostra a imagem atual
    cv2.imshow('Deteccao em Lote', imagem)
    
    
    # waitKey(0) pausa o codigo até que uma tecla seja pressionada
    print("Pressione qualquer tecla para ir para a próxima imagem")
    cv2.waitKey(0) 

cv2.destroyAllWindows()
print("Processamento finalizado")