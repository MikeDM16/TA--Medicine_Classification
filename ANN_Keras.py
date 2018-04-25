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

import csv, os, psutil, time, random, sys, cv2
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
		self.dims = 250 # dimensão das imagens apos tratamento 

		#M_data = M_data[0:5000]
		M_dc = M_data[0:10000]
		M_ref = M_data[10000:len(M_data)]

		# Shuffle the data randomly (keras already allows this doe)
		random.shuffle(M_data)

		M_treino = M_data # train whit all images

		random.shuffle(M_dc)
		#p = int(len(M_dc)*1/3)
		M_teste = M_dc # test only whit 1/3 of DC images

		model = self.treino_progressivo(path_dref, path_dc, M_treino)
		self.save_ANN_model(model)
		
		self.avaliacao(model, path_dc, M_teste)

	# -------------------------------------------------------------------------------------------
	def treino_progressivo(self, path_dref, path_dc, M_data):
		
		# criar uma topologia da rede 
		num_classes = self.num_classes
		epochs = 5 #25 
		model = self.create_compile_model_cnn_plus(num_classes,epochs) 
		print(model.summary()) 

		bloco = len(M_data) # Treinar com bloco de N imagens por fase de treino 
		div = int(len(M_data)/bloco) # quantas vezes é possivel partir o dataset em blocos
		for it in range(0, div):
			# if(it==0):	start = 1 # saltar 1º linha com nome colunas 
			start = it*bloco
			end = start + bloco 
			if(it == div-1): end = len(M_data)
			
			M_imgs, M_target = self.create_ANN_input(path_dc, M_data[start:end])

			print("--- Fase de Treino " + str(it+1) + " ---")
			start_time = time.time()
			self.cnn_simples(model, epochs, M_imgs, M_target)
			
			process = psutil.Process(os.getpid())
			memoryUse = process.memory_info()[0]/2.**20  # memory use in GB...
			print('Memory at use: ' +  str(round(memoryUse, 2)) + " MB.")
			print("Tempo da Fase de Treino " + str(it+1) + ": " + str( round((time.time() - start_time)/60, 2)) + " minutes")
			print("-------------------------")

		return model
	# -------------------------------------------------------------------------------------------
	def progress(self, count, total, status=''):
		bar_len = 50
		filled_len = int(round(bar_len * count / float(total)))

		percents = round(100.0 * count / float(total), 1)
		bar = '=' * filled_len + '-' * (bar_len - filled_len)

		sys.stdout.write('[%s] %s%s    %s\r' % (bar, percents, '%', status))
		sys.stdout.flush() 

	# -------------------------------------------------------------------------------------------
	def avaliacao(self, model, path_dc, M_data):
	
		M_imgs, M_target = self.create_ANN_input(path_dc, M_data)

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
		print(esperados)
		print(predicted)
		correct = (predicted == esperados)
		accuracy = correct.sum() / correct.size
		#accuracy = sklearn.metrics.accuracy_score(esperados, predicted)

		# Calculo RMSE 
		mse = ( ((predicted - esperados)**2).mean() )
		print("Predicted - MSE: " + str(mse))
		print("Predicted - acc: " + str(accuracy))
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	def create_ANN_input(self, path_dc, M_data):
		start_time = time.time()

		# construir conjuntos de treino e teste graduais 
		# M_imgs =  Valores para realizar a aprendizagem
		# M_target =  classe target esperada 
		d1 = len(M_data)
		dims = self.dims

		M_imgs = np.zeros((d1, 3, dims, dims), dtype='uint8')
		M_target = np.zeros((d1,), dtype='uint8')
		
		# Load das imagens efetivamente para memória, na forma de matriz e 
		# com resize, para ocupar menos memória 
		for row in range(0, len(M_data)): 
			''' Imagem de treino no folder ./dc '''
			img_sample = self.load_img(path_dc, M_data[row][1])
			M_imgs[row] = (img_sample) 
			
			M_target[row]= (M_data[row][0])
			self.progress(row, len(M_data), "Loading " + str(len(M_data)) + " images.")
		
		M_target = np.reshape(M_target, (len(M_target), 1))

		if K.image_data_format() == 'channels_last': 
			M_imgs = M_imgs.transpose(0,2,3,1)

		M_imgs = M_imgs.astype('float32')
		M_imgs = M_imgs / 255.0

		process = psutil.Process(os.getpid())
		memoryUse = process.memory_info()[0]/2.**20  # memory use in GB...I think
		print("\nImages loaded: " + str(len(M_imgs)) + ".")
		print('Memory at use: ' +  str(round(memoryUse, 2)) + " MB.")
		print("Tempo Load imagens: " + str( round((time.time() - start_time)/60, 2)) + " minutes")
		print("-------------------------\n")
		return(M_imgs, M_target)
	# -------------------------------------------------------------------------------------------
	
	# -------------------------------------------------------------------------------------------
	def cnn_simples(self, model, epochs, train, test):

		X_train = train
		y_train = keras.utils.to_categorical(test, num_classes=self.num_classes)

		history = model.fit(X_train, y_train, 
							validation_split=0.30, 
							epochs=epochs, 
							batch_size=32, 
							verbose=1)
		
		self.print_history_accuracy(history)
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
		model.add(Conv2D(100, (3, 3), input_shape=(3, dims, dims), padding='same', activation='relu', kernel_constraint=maxnorm(3))) 
		model.add(Dropout(0.2)) 
		model.add(Conv2D(100, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(1000))) 
		model.add(MaxPooling2D(pool_size=(2, 2))) 
		model.add(Flatten()) 
		model.add(Dense(164, activation='relu', kernel_constraint=maxnorm(1000))) 
		model.add(Dropout(0.5)) 
		model.add(Dense(num_classes, activation='softmax')) 
		
		# Compile model 
		lrate = 0.1 
		decay = lrate/epochs 
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
		
		return model
	# -------------------------------------------------------------------------------------------
	def create_compile_model_cnn_plus(self, num_classes,epochs):
		dims = self.dims

		model = Sequential() 
		model.add(Conv2D(32, (3, 3), input_shape=(3, dims, dims), activation='relu', padding='same')) 
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
		model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(1000))) 
		model.add(Dropout(0.2)) 
		model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(1000))) 
		model.add(Dropout(0.2)) 
		model.add(Dense(num_classes, activation='softmax')) 

		# Compile model 
		lrate = 0.01 
		decay = lrate/epochs 
		sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
		return model 
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
		s = self.dims
		img = Image.open(path + str(file))
		
		#plt.imshow(toimage(img)) 
		#plt.show()
		'''
		img1 = resizeimage.resize_cover(img, [250, 250]) # resize
		img2 = resizeimage.resize_cover(img, [100, 100]) # resize
		img3 = resizeimage.resize_cover(img, [50, 50]) # resize		
		_, ax = plt.subplots(2,2, figsize = (12,12))
		ax[0,0].imshow(toimage(img))
		ax[0,0].set_title("Original")
		ax[0,1].imshow(toimage(img1))
		ax[0,1].set_title("250x250")
		ax[1,0].imshow(toimage(img2))
		ax[1,0].set_title("100x100")
		ax[1,1].imshow(toimage(img3))
		ax[1,1].set_title("50x50")
		plt.show()'''

		#img.thumbnail((s, s), Image.ANTIALIAS)
		img = resizeimage.resize_cover(img, [s, s]) # resize
		
		#img = img.convert("RGB")
		img = np.asarray(img) #, dtype=np.float32) / 255
		img = img.reshape(3, s, s) 

		#img =  img[:,:,:3]
		#img = img.transpose(2,0,1)
		
		return img
	# -------------------------------------------------------------------------------------------