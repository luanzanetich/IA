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

# Definir os pares de números para a classificação binária
binary_classes = [3, 8]
binary_train_filter = np.logical_or(y_train == binary_classes[0], y_train == binary_classes[1])
binary_test_filter = np.logical_or(y_test == binary_classes[0], y_test == binary_classes[1])

x_train_binary = x_train[binary_train_filter]
y_train_binary = y_train[binary_train_filter]
x_test_binary = x_test[binary_test_filter]
y_test_binary = y_test[binary_test_filter]

y_train_binary = np.where(y_train_binary == binary_classes[0], 0, 1)
y_test_binary = np.where(y_test_binary == binary_classes[0], 0, 1)

# Dividir os dados de treinamento em treinamento e validação
x_train_binary, x_val_binary, y_train_binary, y_val_binary = train_test_split(
    x_train_binary, y_train_binary, test_size=0.2, stratify=y_train_binary, random_state=42
)

# Definir a arquitetura da CNN binária
model_binary = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar a CNN binária
model_binary.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Definir o caminho para salvar os pesos do modelo binário
checkpoint_path_binary = 'checkpoints/binary_model.ckpt'
checkpoint_binary = ModelCheckpoint(checkpoint_path_binary, save_weights_only=True, save_best_only=True)

# Definir o caminho para salvar o log do treinamento binário
log_path_binary = 'logs/training_log_binary.csv'
csv_logger_binary = CSVLogger(log_path_binary, append=True, separator=',')

# Definir Early Stopping para a classificação binária
early_stopping_binary = EarlyStopping(monitor='val_loss', patience=3)

# Treinar a CNN binária
history_binary = model_binary.fit(
    x_train_binary, y_train_binary, batch_size=128, epochs=10,
    validation_data=(x_val_binary, y_val_binary),
    callbacks=[checkpoint_binary, csv_logger_binary, early_stopping_binary]
)

# Salvar o log de treinamento da classificação binária em um arquivo CSV
history_df_binary = pd.DataFrame(history_binary.history)
history_df_binary.to_csv('logs/training_log_binary.csv', index=False)

# Salvar os hiperparâmetros em um arquivo JSON
hyperparameters_binary = {
    'learning_rate': model_binary.optimizer.lr.numpy().item(),
    'batch_size': 32,
    'epochs': 10
}

with open('hyperparameters_binary.json', 'w') as f:
    json.dump(hyperparameters_binary, f)

# Carregar os melhores pesos salvos durante o treinamento da CNN binária
model_binary.load_weights(checkpoint_path_binary)

# Avaliar a CNN para a classificação binária
loss_binary, accuracy_binary = model_binary.evaluate(x_test_binary, y_test_binary)
print('Binary Test Loss:', loss_binary)
print('Binary Test Accuracy:', accuracy_binary)

# Carregar o log do treinamento binário
binary_log = pd.read_csv('logs/training_log_binary.csv')

# Plotar gráfico de perda ao longo das iterações para a classificação binária
plt.plot(binary_log['loss'], label='Loss - Binary')
plt.plot(binary_log['val_loss'], label='Val Loss - Binary')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Binary Training Loss')
plt.legend()
plt.show()

# Plotar o gráfico de acurácia para o modelo binário
plt.plot(binary_log['accuracy'], label='Accuracy - Binary')
plt.plot(binary_log['val_accuracy'], label='Val Accuracy - Binary')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Binary Training Accuracy')
plt.legend()
plt.show()

##### Reconhecimento de caracteres #####

# Fazer previsões para o conjunto de teste binário
y_pred_binary = model_binary.predict(x_test_binary)
y_pred_binary_classes = np.where(y_pred_binary > 0.5, 1, 0)

#### Cálculo da mtriz de confusão #####

# Criar a matriz de confusão para o modelo binário
cm_binary = confusion_matrix(y_test_binary, y_pred_binary_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Binary')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plotar exemplos de imagens e suas previsões para o modelo binário
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test_binary[i].reshape(28, 28), cmap='gray')
    plt.title('True: {}\nPredicted: {}'.format(y_test_binary[i], y_pred_binary_classes[i]))
    plt.axis('off')
plt.suptitle('Predictions - Binary')
plt.show()

# Salvar o modelo binário
model_binary.save('binary_model.h5')

# Salvar os pesos do modelo binário
with open('binary_model_weights.pkl', 'wb') as f:
    pickle.dump(model_binary.get_weights(), f)

# Salvar as previsões binárias em um arquivo CSV
np.savetxt('output_binary.csv', y_pred_binary_classes, delimiter=',', fmt='%d', header='Output', comments='')
