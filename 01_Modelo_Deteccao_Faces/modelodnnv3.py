import cv2
import numpy as np
import glob 

arquivo_proto = "deploy.prototxt"
arquivo_modelo = "res10_300x300_ssd_iter_140000.caffemodel"

# Carrega a rede neural
print("Carregando modelo DNN")
net = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_modelo)

# Busca todas as imagens jpg dentro da pasta BD
# *.jpg = qualquer arquivo que termine em .jpg
lista_imagens = glob.glob('BD/*.jpg')

# Inicia o loop para processar uma por uma
for caminho_imagem in lista_imagens:
    print(f"Processando: {caminho_imagem}")
    
    imagem = cv2.imread(caminho_imagem)
    
    # Verificação de segurança
    if imagem is None:
        print(f"Erro ao abrir {caminho_imagem}")
        continue 

    # Pega altura e largura para os cálculos depois
    (h, w) = imagem.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Passa o blob pela rede
    net.setInput(blob)
    deteccoes = net.forward()

    # Analisa as detecções encontradas
    for i in range(0, deteccoes.shape[2]):
        # Pega a confiança
        confianca = deteccoes[0, 0, i, 2]

        # Só aceita se tiver mais de 50% de certeza
        if confianca > 0.5:
            # Calcula a posição do quadrado (box)
            box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Desenha o retângulo e o texto
            texto = "{:.2f}%".format(confianca * 100)
            
            # Ajuste para o texto não sair da tela
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(imagem, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(imagem, texto, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    

    cv2.imshow('Deteccao DNN em Lote', imagem)
    
    print("Pressione qualquer tecla para ir para a próxima imagem")
    cv2.waitKey(0) 

cv2.destroyAllWindows()
print("Processamento finalizado")