import numpy as np 
import keras 
from keras.datasets import cifar10 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.layers import Flatten 
from keras.constraints import maxnorm
from keras.optimizers import SGD 
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D 
from keras.utils import np_utils
from keras import backend as K 
K.set_image_dim_ordering('th') #pode ser 'th' ou 'tf' 
import matplotlib.pyplot as plt
from scipy.misc import toimage

import csv, os, psutil, time, random
from matplotlib import pylab
from scipy.misc import toimage
from PIL import Image, ImageColor, ImageTk
from resizeimage import resizeimage

# fixar random seed para se puder reproduzir os resultados 
seed = 9 
np.random.seed(seed) 

class ANN_Keras():
	def __init__(self, path_dref, path_dc, M_data):
		self.num_classes = 1000  # número de classes
		self.dims = 32 # dimensão das imagens apos tratamento 

		model = self.treino_progressivo(path_dref, path_dc, M_data)
		self.save_ANN_model(model)
		
		self.avaliacao(model, path_dc, M_data)

	# -------------------------------------------------------------------------------------------
	def treino_progressivo(self, path_dref, path_dc, M_data):
		# Shuffle the data randomly (keras already allows this doe)
		random.shuffle(M_data)

		# criar uma topologia da rede 
		num_classes = self.num_classes
		epochs = 3 #25 
		model = self.create_compile_model_cnn_simples(num_classes,epochs) 
		print(model.summary()) 

		bloco = 10 # Treinar com bloco de N imagens por fase de treino 
		div = int(len(M_data)/bloco) # quantas vezes é possivel partir o dataset em blocos
		for it in range(0, div):
			# if(it==0):	start = 1 # saltar 1º linha com nome colunas 
			start = it*bloco
			end = start + bloco 
			
			M_imgs, M_target = self.create_ANN_input(path_dc, start, end, M_data)
			
			print("--- Fase de Treino " + str(it+1) + " ---")
			start_time = time.time()
			self.cnn_simples(model, epochs, M_imgs, M_target)
			
			process = psutil.Process(os.getpid())
			memoryUse = process.memory_info()[0]/2.**20  # memory use in GB...I think
			print("Images load: " + str(len(M_imgs)) + " Train samples.")
			print('Memory at use: ' +  str(round(memoryUse, 2)) + " MB.")
			print("Tempo da Fase de Treino " + str(it+1) + ": " + str( round((time.time() - start_time)/60, 2)) + " minutes")
			print("-------------------------")

		return model
	# -------------------------------------------------------------------------------------------
	
	# -------------------------------------------------------------------------------------------
	def avaliacao(self, model, path_dc, M_data):
		# Shuffle the data randomly (keras already allows this doe)
		random.shuffle(M_data)
		p = int(len(M_data)*2/3)
		M_imgs, M_target = self.create_ANN_input(path_dc, 0, p, M_data)

		X_test = M_imgs
		y_test = keras.utils.to_categorical(M_target, self.num_classes)
		
		# Avaliação final com os casos de teste 
		print("--- Evaluate function scores --- ")
		scores = model.evaluate(X_test, y_test, verbose=1)  
		print("Accuracy: %.2f%%" % (scores[1]*100)) 
		print("Erro modelo CNN simples: %.2f%%" % (100-scores[1]*100))
		for i in range(0, len(scores)):
			print("Evaluate - " + model.metrics_names[i] + ": " + str(scores[i]))


		# --- Avaliação desempenho segundo função Predict 
		print("--- Predict function scores --- ")
		predicted = model.predict(X_test)
		predicted = [np.argmax(row) for row in predicted]
		print(len(predicted))

		# Adaptção dos dados para np array
		predicted = np.asarray(predicted, dtype=float)
		esperados = np.asarray(M_target, dtype=float)
		
		correct = (predicted == esperados)
		accuracy = correct.sum() / correct.size
		#accuracy = sklearn.metrics.accuracy_score(esperados, predicted)

		# Calculo RMSE 
		mse = ( ((predicted - esperados)**2).mean() )
		print("Predicted - MSE: " + str(mse))
		print("Predicted - acc: " + str(accuracy))
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	def create_ANN_input(self, path_dc, start, end, M_data):
		# construir conjuntos de treino e teste graduais 
		M_imgs = []  # Valores para realizar a aprendizagem
		M_target = [] # classe target esperada 

		# Load das imagens efetivamente para memória, na forma de matriz e 
		# com resize, para ocupar menos memória 
		for row in range(start, end): 
			''' Imagem de treino no folder ./dc '''
			img_sample = self.load_img(path_dc, M_data[row][1])
			M_imgs.append(img_sample);
			#''' Imagem de referência no folder ./dr '''
			# img_target = load_img(path_dref, M_data[row][0])
			#M_target.append(img_target);
				
			M_target.append(M_data[row][0])
		
		M_imgs = np.asarray(M_imgs)
		M_target = np.asarray(M_target)

		return(M_imgs, M_target)
	# -------------------------------------------------------------------------------------------
	
	# -------------------------------------------------------------------------------------------
	def cnn_simples(self, model, epochs, train, test):

		X_train = train
		y_train =  keras.utils.to_categorical(test, self.num_classes)
	
		history=model.fit(X_train, y_train, 
							validation_data=0, 
							epochs=epochs, 
							batch_size=32, 
							verbose=1)
		#self.print_history_accuracy(history)
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	'''
	- camada convolucional de entrada com 32 feature maps de tamanho 3×3, com activação por 
	rectifier (relu) e restrição aos pesos de max norm a 3
	- Dropout em 20%. 
	- camada convolucional com 32 feature maps de tamanho 3×3, com activação por rectifier 
	(relu) e restrição aos pesos de max norm a 3
	- Camada Max Pool com tamanho 2×2. 
	- Camada Flatten. 
	- Camada completamente ligada com 512 neuronios e uma fução de activação 'rectifier 
	activation function'. 
	- Dropout em 50%. 
	- Camada de saída completamente ligada com 10 neuronios e função de activação softmax. 
	- O modelo é treinado utilizando logarithmic loss e o algoritmo de gradient descent é o
	SGD (Stochastic gradient descent optimizer) com um valor alto de momentum e queda nos 
	pesos, começando com uma taxa de aprendizagem de 0.01:
	Adicionalmente foi acrescentado uma restrição nos pesos de cada camada garantindo assim que	a norma maxima dos pesos nao excede o valor de 3.
	Isto consegue-se colocando o parametro kernel_constraint na classe Dense igual a 3. '''
	# -------------------------------------------------------------------------------------------
	def create_compile_model_cnn_simples(self, num_classes, epochs):
		dims = self.dims

		model = Sequential() 
		model.add(Conv2D(20, (3, 3), input_shape=(3, dims, dims), padding='same', activation='relu', kernel_constraint=maxnorm(3))) 
		model.add(Dropout(0.2)) 
		model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))) 
		model.add(MaxPooling2D(pool_size=(2, 2))) 
		model.add(Flatten()) 
		model.add(Dense(164, activation='relu', kernel_constraint=maxnorm(3))) 
		model.add(Dropout(0.5)) 
		model.add(Dense(num_classes, activation='softmax')) 
		
		# Compile model 
		lrate = 0.01 
		decay = lrate/epochs 
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
		
		return model
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	# utils para visulaização do historial de aprendizagem
	# Só podem ser utilizadas caso se use percentage split na fase de treino.
	
	# como o treino é gradual, isto não acontece!!! 

	# A RNA é avaliada com a função predict e a função evaluate, no fim de todas as fases de treino 
	# iterativas 

	def print_history_accuracy(self, history):
		print(history.history.keys())
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
	
	def print_history_loss(self, history):
		print(history.history.keys())
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()
	# -------------------------------------------------------------------------------------------
	def save_ANN_model(self, model):
		# Guardar a rede criada (topologia + peses ligações)
		model_json = model.to_json()
		with open('model.json', 'w') as json_file:
			json_file.write(model_json)

		model.save_weights('model.h5')

	def read_ANN_model(self):
		# read in your saved model structure
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		# and create a model from that
		model = model_from_json(loaded_model_json)
		# and weight your nodes with your saved values
		model.load_weights('model.h5')
		return model

	# -------------------------------------------------------------------------------------------
	# Função que realizar o resize de uma imagem 
	# Resize images allows to reduce needed RAM to store the data for the ANN 
	def resize_size(self, size): 
		w, h = size
		# h/w  =  500/x   --> x = 500*w/h
		new_h = self.dims # resize to an ratio with predifined px of height
		max_size = (int(new_h*w/h), new_h)
		return max_size
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	# Função que carrega uma imagem para uma matriz, através do nome do ficheiro e da diretoria 
	# Realiza o resize da imagem para um proporção pré definida e troca os canais 
	def load_img(self, path, file):
		img = Image.open(path + str(file))
		s = self.dims
		img = resizeimage.resize_cover(img, [s, s]) # resize
		img = img.convert("RGB")
		img = np.asarray(img, dtype=np.float32) / 255
		img =  img[:,:,:3]
		img = img.transpose(2,0,1)
		return img
	# -------------------------------------------------------------------------------------------