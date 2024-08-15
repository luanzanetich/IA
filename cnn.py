import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import json
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import copy

# Verificar se o diretório 'logs' existe e criá-lo se necessário
logs_dir = 'logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Pré-processamento dos dados
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train_categorical = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Dividir os dados em conjuntos de treinamento e validação
x_train, x_val, y_train_categorical, y_val_categorical = train_test_split(x_train, y_train_categorical, test_size=0.2, random_state=42)

# Definir a arquitetura da CNN multiclasse
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilar a CNN multiclasse
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definir o caminho para salvar os pesos do modelo multiclasse
checkpoint_path = 'checkpoints/multiclass_model.ckpt'
checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True)

# Definir o caminho para salvar o log do treinamento
log_path = 'logs/training_log.csv'
csv_logger = CSVLogger(log_path, append=True, separator=',')

# Definir Early Stopping para a classificação multiclasse
early_stopping_multiclass = EarlyStopping(monitor='val_loss', patience=3)

# Treinar a CNN multiclasse
history_multiclass = model.fit(x_train, y_train_categorical, batch_size=128, epochs=10, validation_data=(x_val, y_val_categorical), callbacks=[checkpoint, csv_logger, early_stopping_multiclass])

# Salvar o log de treinamento da classificação multiclasse em um arquivo CSV
history_df_multiclass = pd.DataFrame(history_multiclass.history)
history_df_multiclass.to_csv('logs/training_log_multiclass.csv', index=False)

# Salvar os hiperparâmetros em um arquivo JSON
hyperparameters = {
    'learning_rate': model.optimizer.lr.numpy().item(),
    'batch_size': 32,
    'epochs': 10
}

with open('hyperparameters_multiclass.json', 'w') as f:
    json.dump(hyperparameters, f)

# Carregar os melhores pesos salvos durante o treinamento da CNN multiclasse
model.load_weights(checkpoint_path)

# Avaliar a CNN para a classificação multiclasse
loss, accuracy = model.evaluate(x_test, y_test_categorical)
print('Multiclass Test Loss:', loss)
print('Multiclass Test Accuracy:', accuracy)

# Carregar o log do treinamento multiclasse
multiclass_log = pd.read_csv('logs/training_log_multiclass.csv')

# Plotar gráfico de perda ao longo das iterações para a classificação multiclasse
plt.plot(range(1, len(multiclass_log) + 1), multiclass_log['loss'], label='Loss - Multiclass')
plt.plot(range(1, len(multiclass_log) + 1), multiclass_log['val_loss'], label='Val Loss - Multiclass')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Multiclass Training Loss')
plt.legend()
plt.show()

# Plotar gráfico de acurácia ao longo das iterações para a classificação multiclasse
plt.plot(range(1, len(multiclass_log) + 1), multiclass_log['accuracy'], label='Accuracy - Multiclass')
plt.plot(range(1, len(multiclass_log) + 1), multiclass_log['val_accuracy'], label='Val Accuracy - Multiclass')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Multiclass Training Accuracy')
plt.legend()
plt.show()

# Fazer previsões para o conjunto de teste multiclasse
y_pred_multiclass = model.predict(x_test)
y_pred_multiclass = np.argmax(y_pred_multiclass, axis=1)

# Criar a matriz de confusão para o modelo multiclasse
cm_multiclass = confusion_matrix(y_test, y_pred_multiclass)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_multiclass, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Multiclass')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plotar exemplos de imagens e suas previsões para o modelo multiclasse
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('True: {}\nPredicted: {}'.format(y_test[i], y_pred_multiclass[i]))
    plt.axis('off')
plt.suptitle('Predictions - Multiclass')
plt.show()



# Salvar os modelos treinados
model.save('multiclass_model.h5')

# Salvar os pesos dos modelos em formato pickle
with open('multiclass_model_weights.pkl', 'wb') as f:
    pickle.dump(model.get_weights(), f)
   
# Salvar as saídas produzidas pela rede neural para cada um dos dados de teste
y_pred_multiclass = model.predict(x_test)

# Mapear as saídas para as classes correspondentes
y_pred_multiclass_classes = np.argmax(y_pred_multiclass, axis=1)

# Salvar as saídas produzidas pela rede neural para cada um dos dados de teste
np.savetxt('output_multiclass.csv', y_pred_multiclass_classes, delimiter=',', fmt='%d')
