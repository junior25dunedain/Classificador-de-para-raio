import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as sts
import scikitplot as skplt
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def Carrega_dados():
    data = pd.read_csv('corrente_PR_BOM.csv',header=None)
    data2 = pd.read_csv('corrente_PR_DEF.csv',header=None)

    lista2 = []
    lista4 = []

    for j in range(40): # sub-amostragem para equilibrar os dados das 2 fontes BD
        lista1 = []
        lista3 = []
        for k in range(200):
            if k == 0:
                lista1.append(sts.mean(data.loc[j,0:249]))
                lista3.append(sts.mean(data2.loc[j,0:49]))
            elif k == 199:
                lista1.append(sts.mean(data.loc[j,49750:49999]))
                lista3.append(sts.mean(data2.loc[j, 9950:9999]))
            else:
                lista1.append(sts.mean(data.loc[j,((249*k)+k) : (((k+1)*249)+k)]))
                lista3.append(sts.mean(data2.loc[j, ((49 * k) + k): (((k + 1) * 49) + k)]))

        lista2.append(lista1)
        lista4.append(lista3)
    bom = pd.DataFrame(lista2,dtype= 'float64')
    ruim = pd.DataFrame(lista4,dtype= 'float64')

    cont = 1    # pega 20 elementos aleatorios entre os dados bons e os ruins, para formar a base de dados de entradas (40,200)
    lista_ind = []
    bom2 = []
    ruim2 = []
    while cont <= 20:
        j = random.randint(0, 39)

        if j in lista_ind:
            continue
        lista_ind.append(j)
        bom2.append(bom.loc[j])
        ruim2.append(ruim.loc[j])
        cont += 1
    entradas = pd.concat([pd.DataFrame(bom2), pd.DataFrame(ruim2)], ignore_index=True)

    alvo_bom = pd.DataFrame(np.zeros((20,1))) # criação da base de dados alvo (40,1)
    alvo_ruim = pd.DataFrame(np.ones((20,1)))
    alvos = pd.concat([alvo_bom, alvo_ruim],ignore_index=True)

    entradas = np.asarray(entradas,dtype= 'float64')
    alvos = np.asarray(alvos, dtype='float64')

    return entradas, alvos

def Saida_aleatoria(Ent_test,mod_treinado):
    x = random.randint(0,len(Ent_test)-1)
    y = mod_treinado.predict(Ent_test)
    if y[x] == 1:
        print('O Para-raio está defeituoso !!')
    else:
        print('O Para-raio está bom !!')


# main
entradas , alvos = Carrega_dados()

#crias as bases de treinamento e teste
Ent_tre, Ent_test, Alvo_tre, Alvo_test = train_test_split(entradas,alvos,test_size=0.3, random_state=42, stratify= alvos)

# cria o modelo neural ( com uma unica camada de 101 neuronios)
net = MLPClassifier(activation= 'identity', solver='lbfgs', max_iter=300, hidden_layer_sizes=(101),verbose=True)

# realiza o treinamento do modelo neural
modelo_class_QEE = net.fit(Ent_tre,Alvo_tre)

# estima a precisão do modelo treinado
score = modelo_class_QEE.score(Ent_test, Alvo_test)
print('\n',f' A precisão é {score*100}%', '\n')

# calcula as previsoes do modelo
previsoes = modelo_class_QEE.predict(Ent_test)
prevpb = modelo_class_QEE.predict_proba(Ent_test)

print(classification_report(Alvo_test, previsoes))

# usando a biblioteca scikitplot
skplt.metrics.plot_confusion_matrix( Alvo_test, previsoes)
plt.show()
skplt.metrics.plot_confusion_matrix(Alvo_test, previsoes, normalize='True')
plt.show()

# plot a ROC
skplt.metrics.plot_roc(Alvo_test,prevpb)
plt.show()

# resultado aleatorio do modelo de rede neural MLP
Saida_aleatoria(Ent_test,modelo_class_QEE)