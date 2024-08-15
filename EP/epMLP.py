# - ALGORITMO MLP -
# - problema de identificação de caracteres -
#       63 neuronios na camada de entrada
#       10 neuronios na camada escondida
#       7 neuronios na camada de saida
#       função de ativaçao: tangente hiberbolica

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def activation_function(t):
    return (np.exp(2*t)-1)/(np.exp(2*t)+1)

def derivada_activation(f):
    return (1/np.cosh(f)**2)

def mlp_architecture(input_length, hidden_length, output_length):

# pesos iniciais de forma aleatoria - valores entre -0.5 e 0.5
    weights_hidden = np.random.uniform(-0.5, 0.5, size=(hidden_length, input_length))
    weights_output = np.random.uniform(-0.5, 0.5, size=(output_length, hidden_length))

    return weights_hidden, weights_output

def mlp_forward(input, hidden, output, weights_hidden, weights_output):

# combinação linear de pesos da hidden e inputs + função de ativação
    hidden = np.dot(weights_hidden, input)
    hiddenActivation = activation_function(hidden)

# combinação linear dos pesos entre hidden e length e valores dos neuronios da hidden + função de ativação  
    output = np.dot(weights_output, hiddenActivation)
    outputFinal = activation_function(output)

    return hidden, output, hiddenActivation, outputFinal, weights_hidden, weights_output

def mlp_backpropagation(input, output, outputFinal, target, hidden, hiddenActivation, weights_hidden, weights_output, learningRate):

# cálculo do delta da camada de saída
    deltaOutput = (target - output) * derivada_activation(output)

# cálculo do delta da camada escondida
    deltaHidden = np.dot(deltaOutput, weights_output) * derivada_activation(hidden)

# atualização dos pesos 
    weights_output += learningRate * np.outer(deltaOutput, hiddenActivation)
    weights_hidden += learningRate * np.outer(deltaHidden, input)

    return output, outputFinal, hidden, weights_hidden, weights_output

def extracaoDados():

# separação dos dados de forma aleatoria na proporção 2/3 para treinamento e 1/3 para teste
    df1 = pd.read_csv('caracteres-ruido.csv', header=None)
    df2 = pd.read_csv('caracteres_ruido20.csv', header=None)
    df3 = pd.read_csv('caracteres-limpo.csv', header=None)
    allData = pd.concat([df1, df2, df3])
    allData = allData.sample(frac=1).reset_index(drop=True)
    train_size = int(2/3 * len(allData))
    train_data = allData.iloc[:train_size, :]
    test_data = allData.iloc[train_size:, :]

# Separar as entradas e os targets dos conjuntos de treinamento e teste
    inputs = train_data.iloc[:, :63].values
    targets = train_data.iloc[:, -7:].values
    inputsTeste = test_data.iloc[:, :63].values
    targetsTeste = test_data.iloc[:, -7:].values

    return inputs, targets, inputsTeste, targetsTeste

def treinamento(epochs, inputs, targets, input, hidden, output, weights_hidden, weights_output, learningRate, maxError, dados_errosIteracoes):
    for i in range(epochs):
        error = 0
        for j in range(len(inputs)):
            input = inputs[j]
            target = targets[j]
            hidden, output, hiddenActivation, outputFinal, weights_hidden, weights_output = mlp_forward(input, hidden, output, weights_hidden, weights_output)
            output, outputFinal, hidden, weights_hidden, weights_output = mlp_backpropagation(input, output, outputFinal, target, hidden, hiddenActivation, weights_hidden, weights_output, learningRate)

            error += 1/len(inputs) * np.sum((target - outputFinal)**2)
            if error <= maxError:
                break
        print('ERRO QUADRATICO MEDIO da epoca ', i,': ', error)
        dados_errosIteracoes.append([i, error])

    return dados_errosIteracoes, weights_hidden, weights_output
    
def testes(inputsTeste, targetsTeste, hidden_length, output_length, weights_hidden, weights_output, dados_outputs, indicesEsperados, indicesPrevistos, rightAns):
    for i in range(len(inputsTeste)):
        input = inputsTeste[i]
        target = targetsTeste[i]
        hidden = np.zeros(hidden_length)
        output = np.zeros(output_length)
        hidden, output, hiddenActivation, outputFinal, wsHidden, wsOutput = mlp_forward(input, hidden, output, weights_hidden, weights_output)
            
    # pega-se os indices do elemento mais alto de cada array, para verificar se a resposta está correta
        indEsperado = np.argmax(outputFinal)
        indPrevisto = np.argmax(target)

        indicesEsperados.append(indEsperado)
        indicesPrevistos.append(indPrevisto)
            
        print('Entrada:', input)
        print('Target:', target)
        print('Saida:', outputFinal)

        dados_outputs.append([input, target, outputFinal])

    # verificação da resposta
        if indPrevisto == indEsperado:
            rightAns += 1
    return dados_outputs, indicesEsperados, indicesPrevistos, rightAns

def paraExcel(dados_hiperparametros, weights_hidden, weights_output, dados_errosIteracoes, dados_outputs, matrizConfusaoGeral, dados_mediaDesv, df_pesosIniciaisHidden, df_pesosIniciaisOutput):
    df_hiperparametros = pd.DataFrame(dados_hiperparametros, columns= ["epoca", "taxa de aprendizado", "maxError", "input_length", "hidden_length", "output_length"])
    df_pesosFinaisHidden = pd.DataFrame(weights_hidden)
    df_pesosFinaisOutput = pd.DataFrame(weights_output)
    df_errosIteracoes = pd.DataFrame(dados_errosIteracoes, columns=["epoca", "erro"])
    df_outputs = pd.DataFrame(dados_outputs, columns= ["input", "target", "output"])
    df_matrizConfusao = pd.DataFrame(matrizConfusaoGeral)
    df_mediaDesv = pd.DataFrame(dados_mediaDesv, columns= ["media", "desvio Padrao"])

    with pd.ExcelWriter('mlp_dados.xlsx') as writer:
        df_hiperparametros.to_excel(writer, sheet_name='hiperparametros', index=False)
        df_pesosIniciaisHidden.to_excel(writer, sheet_name='pesos iniciais hidden', index=False)
        df_pesosIniciaisOutput.to_excel(writer, sheet_name='pesos iniciais output', index=False)
        df_pesosFinaisHidden.to_excel(writer, sheet_name='pesos finais hidden', index=False)
        df_pesosFinaisOutput.to_excel(writer, sheet_name='pesos finais output', index=False)
        df_errosIteracoes.to_excel(writer, sheet_name='errosIteracoes', index=False)
        df_outputs.to_excel(writer, sheet_name='outputs', index=False)
        df_matrizConfusao.to_excel(writer, sheet_name='matrizConfusao', index=False)
        df_mediaDesv.to_excel(writer, sheet_name='media e desvio Padrao', index=False)

def main():

# arrays para formação do arquivo CSV
    dados_hiperparametros = []
    dados_errosIteracoes = []
    dados_outputs = []
    dados_mediaDesv = []

# hiperparametros iniciais
    learningRate = 0.2
    epochs = 10000
    maxError = 0.001
    input_length = 63
    hidden_length = 30
    output_length = 7
    dados_hiperparametros.append([epochs, learningRate, maxError, input_length, hidden_length, output_length])
    hidden = np.zeros(hidden_length)
    output = np.zeros(output_length)

# chamada à função inicial
    weights_hidden, weights_output = mlp_architecture(input_length, hidden_length, output_length)

    pesosIniciaisHidden = np.copy(weights_hidden)
    pesosIniciaisOutput = np.copy(weights_output)

# criação de dataframes para criação posterior do excel
    df_pesosIniciaisHidden = pd.DataFrame(pesosIniciaisHidden)
    df_pesosIniciaisOutput = pd.DataFrame(pesosIniciaisOutput)

#criação de arrays para guardar os indices (esperados e previstos) para a matriz de confusão
    todosIndicesEsperados = []
    todosIndicesPrevistos = []

#array com as respostas dos testes
    resposta = []

# HOLDOPUT - 5 iterações treinamento e teste -> posteriormente calcula-se a média e desvio padrão dessas 5 iterações
    for _ in range(5):
        inputs, targets, inputsTeste, targetsTeste = extracaoDados()
        indicesEsperados = []
        indicesPrevistos = []
        rightAns = 0        

        dados_errosIteracoes, weights_hidden, weights_output = treinamento(epochs, inputs, targets, input, hidden, output, weights_hidden, weights_output, learningRate, maxError, dados_errosIteracoes)
        dados_outputs, indicesEsperados, indicesPrevistos, rightAns = testes(inputsTeste, targetsTeste, hidden_length, output_length, weights_hidden, weights_output, dados_outputs, indicesEsperados, indicesPrevistos, rightAns)

        resposta.append(rightAns)
        todosIndicesEsperados.extend(indicesEsperados)
        todosIndicesPrevistos.extend(indicesPrevistos)
    print(resposta)
    dados_mediaDesv.append([np.mean(resposta), np.std(resposta)])

# matriz de confusão com todos os testes (das 5 iterações do holdout)
    matrizConfusaoGeral = confusion_matrix(todosIndicesEsperados, todosIndicesPrevistos)
    print(matrizConfusaoGeral)

# chamada à função para criar o arquivo excel que contém: hiperparâmetros, pesos iniciais e finais, erros de cada iteração, outputs de cada iteração, matriz de confusão e média e desvio padrão
    paraExcel(dados_hiperparametros, weights_hidden, weights_output, dados_errosIteracoes, dados_outputs, matrizConfusaoGeral, dados_mediaDesv, df_pesosIniciaisHidden, df_pesosIniciaisOutput)

main()