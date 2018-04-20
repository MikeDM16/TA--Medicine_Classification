import numpy as np 
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
		start_time = time.time()

		''' Resize images allows to reduce needed RAM to store the data '''  
		def resize_size(size): 
			w, h = size
			# h/w  =  500/x   --> x = 500*w/h
			new_h = 500 # resize to an ratio with 500px of height
			max_size = (int(new_h*w/h), new_h)
			return (50, 50)

		# Shuffle the data randomly (keras already allows this doe)
		random.shuffle(M_data)

		'''# Treinar com N imagens por fase de treino 
		bloco = 250
		div = int(len(M_data)/bloco)
		for it in range(0, div):
			start = it*div
			treino = M_data[start: start+div]
		'''
		
		# construir conjuntos de treino e teste graduais 
		M_imgs = []
		M_target = []

		def load_img(path, row, opt):
			img = Image.open(path + str(M_data[row][opt]))
			#max_size = resize_size(img.size)
			img = resizeimage.resize_cover(img, [100, 100]) # resize
			img = img.convert("RGB")
			img = np.asarray(img, dtype=np.float32) / 255
			img =  img[:,:,:3]
			img = img.transpose(2,0,1)
			return img

		# Saltar 1º linha com nome das colunas
		for row in range(1, 51): 
			''' Imagem de treino no folder ./dc '''
			img_sample = load_img(path_dc, row, 1)
			''' Imagem de referência no folder ./dr '''
			img_target = load_img(path_dref, row, 0)

			M_imgs.append(img_sample);
			M_target.append(img_target);
		
		print("Samples Treino: " + str(len(M_imgs)))
		self.cnn_simples(M_imgs, M_target)

		#plt.imshow(toimage(img_sample)) 
		#plt.show()

		process = psutil.Process(os.getpid())
		memoryUse = process.memory_info()[0]/2.**20  # memory use in GB...I think
		print("Images load: " + str(len(M_imgs)) + " Train samples, " + str(len(M_target)) + " Target samples.")
		print('Memory use: ' +  str(round(memoryUse, 2)) + " MB.")
		print("Time to load images: " + str( round((time.time() - start_time)/60, 2)) + " minutes")


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
	# Etapa 2 - Definir a topologia da rede (arquitectura do modelo) e compilar
	def create_compile_model_cnn_simples(self, num_classes, epochs):
		model = Sequential() 
		model.add(Conv2D(100, (3, 3), input_shape=(3, 100, 100), padding='same', activation='relu', kernel_constraint=maxnorm(3))) 
		model.add(Dropout(0.2)) 
		'''model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3))) 
		model.add(MaxPooling2D(pool_size=(2, 2))) '''
		model.add(Flatten()) 
		model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
		model.add(Dropout(0.5)) 
		model.add(Dense(num_classes, activation='softmax')) 
		
		# Compile model 
		lrate = 0.01 
		decay = lrate/epochs 
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
		
		return model

	'''' Etapa 2 - Definir a topologia da rede (arquitectura do modelo) e compilar
	def create_compile_model_cnn_plus(self, num_classes,epochs):
		model = Sequential() 
		model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same')) 
		model.add(Dropout(0.2)) 
		model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
		model.add(MaxPooling2D(pool_size=(2, 2))) 
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
		model.add(Dropout(0.2)) 
		model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
		model.add(MaxPooling2D(pool_size=(2, 2))) 
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
		model.add(Dropout(0.2)) 
		model.add(Conv2D(128, (3, 3), activation='relu', padding='same')) 
		model.add(MaxPooling2D(pool_size=(2, 2))) 
		model.add(Flatten()) 
		model.add(Dropout(0.2)) 
		model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3))) 
		model.add(Dropout(0.2)) 
		model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3))) 
		model.add(Dropout(0.2)) 
		model.add(Dense(num_classes, activation='softmax')) 
		
		# Compile model 
		lrate = 0.01 
		decay = lrate/epochs 
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
		
		return model 
	'''

	def cnn_simples(self, train, test):
		p = int(len(train)/2)

		train = np.asarray(train)
		test = np.asarray(test)
		
		(X_train, y_train) = train[0:p], train[p:len(train)] 
		(X_test, y_test) = test[0:p], test[p:len(test)]

		# normalize inputs from 0-255 to 0.0-1.0 
		#X_train = X_train.astype('float32') #converter de inteiro para real 
		#X_test = X_test.astype('float32') 
		#X_train = X_train / 255.0 
		#X_test = X_test / 255.0 
		# transformar o label que é um inteiro em categorias binárias, o valor passa a ser o  correspondente à posição 
		# a classe 5 passa a ser a lista [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] 

		#plt.imshow(toimage(y_train[1])) 
		#plt.show()

		#y_train = np_utils.to_categorical(y_train)
		#y_test = np_utils.to_categorical(y_test)

		num_classes = y_test.shape[0] 
		
		epochs = 5 #25 
		model = self.create_compile_model_cnn_simples(num_classes,epochs) 
		print(model.summary()) 

		print("shape X_train" + str(X_train.shape))
		print("shape X_teste" + str(X_test.shape))
		print("shape y_train" + str(y_train.shape))
		print("shape y_teste" + str(y_test.shape))

		# treino do modelo: epochs=5, batch size = 32 
		history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, verbose=2)
		print_history_accuracy(history) 

		# Avaliação final com os casos de teste 
		scores = model.evaluate(X_test, y_test, verbose=0) 
		print('Scores: ', scores) 
		print("Accuracy: %.2f%%" % (scores[1]*100)) 
		print("Erro modelo CNN cifar10 simples: %.2f%%" % (100-scores[1]*100))

	'''def cnn_plus():
		(X_train, y_train), (X_test, y_test) = cifar10.load_data() 
		#(X_train, y_train), (X_test, y_test) = load_cfar10_dataset() 
		# normalize inputs from 0-255 to 0.0-1.0 
		X_train = X_train.astype('float32') #converter de inteiro para real 
		X_test = X_test.astype('float32') 
		X_train = X_train / 255.0 
		X_test = X_test / 255.0 
		# transformar o label que é um inteiro em categorias binárias, o valor passa a ser o correspondente à posição 
		# a classe 5 passa a ser a lista [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] 
		y_train = np_utils.to_categorical(y_train) 
		y_test = np_utils.to_categorical(y_test) 
		num_classes = y_test.shape[1] 
		epochs = 5 #25 
		model = self.create_compile_model_cnn_plus(num_classes,epochs) 
		print(model.summary()) 

		# treino do modelo: epochs=5, batch size = 64 
		history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, 
		batch_size=64, verbose=2)
		print_history_accuracy(history) 

		# Avaliação final com os casos de teste 
		scores = model.evaluate(X_test, y_test, verbose=0) 
		print('Scores: ', scores) 
		print("Accuracy: %.2f%%" % (scores[1]*100)) 
		print("Erro modelo CNN cifar10 simples: %.2f%%" % (100-scores[1]*100))'''
