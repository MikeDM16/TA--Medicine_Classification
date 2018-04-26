import csv, os, psutil, time, sys, pathlib
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt
from scipy.misc import toimage
from PIL import Image, ImageColor, ImageTk
from keras.preprocessing.image import ImageDataGenerator, load_img,  array_to_img, img_to_array

class Tratamento():
	def tratamento_classes(self, file_name, path_dref, path_dc):
			
		def load_dados(file_name, path_dref, path_dc):
			# matriz com os dados do CSV groundTruthTable inicial
			M_data = self.read_csv_file(file_name)

			# Aumentar os casos de treino com as imagens de referência
			M_data = self.adicionar_imgs_ref(path_dref, path_dc, M_data)

			# Matriz com o mapeamento entre o nome das cápsulas e um id da classe atribuido
			ref_classes = self.create_ids_classes(path_dref)

			M_data = self.troca_nomes_ids(ref_classes, M_data)
			M_data = M_data[1:len(M_data)]

			M_data = self.adicionar_imgs_DataAugmentation(path_dref, path_dc, M_data)
			
			return M_data

		# Executar isto só 1 vez. Tendo as imagens preparadas não há necessidade de repetir isto
		#M_data = load_dados(file_name, path_dref, path_dc)
		M_data = self.read_csv_file("Dados_final.csv")
		
		#self.Data_Augmentation(path_dref, path_dc, M_data)
		#M_data = self.adicionar_imgs_DataAugmentation(path_dref, path_dc, M_data)

		return M_data
	# -------------------------------------------------------------------------------------------
	def adicionar_imgs_DataAugmentation(self, path_dref, path_dc, M_data):
		path_da = "./DataAugmentation/"
		da_imgs = os.listdir(path_da)

		nome = 7000

		M_aux = [] 
		
		aux = np.zeros((1000))

		i=0;
		while(i < len(da_imgs)):
			# abrir img em ./DataAugmentation
			img = Image.open(path_da +  da_imgs[i])
			# Copiar img para ./dc com novo nome 
			new_file_name =  str(nome) + ".jpeg"
			img.save(path_dc + new_file_name)

			# O nome das imagens teve um bug. 
			# Para saber a classe precisa desta conta
			# é mais facil improvisar aqui que perder horas a criar novas imgs
			classe = int(da_imgs[i].split("_")[0])
			
			aux[classe-1] = 1; 

			# Adicionar os novos casos treino à matriz principal
			M_data.append([str(classe), new_file_name])			
			M_aux.append([str(classe), new_file_name])			
			
			i += 1
			nome += 1

			text = "Copying " + str(len(da_imgs)) + " Data augmentation images."
			self.progress(i, len(da_imgs), text) 

		self.save_csv(M_aux, "imgs_dataAugmentation.csv")
		self.save_csv(M_data, "Dados_final.csv")
		print(aux.sum())
		print(i)
		
		return M_data

	# -------------------------------------------------------------------------------------------
	# Cada imagem de referência (target) será adicionada ao folder DC 
	def adicionar_imgs_ref(self, path_dref, path_dc, M_data):
		# Obter o nome de todas as imagens de referência. 
		ref_imgs = os.listdir(path_dref)
		dc_imgs = os.listdir(path_dc)

		# As imagens no folder dc estão numeradas entre [0,4999]
		# As imagens que se coloquem lá têm que começar por 5000.jpg
		#nome = len(dc_imgs)
		nome = 5000
		i=0;
		while(i < len(ref_imgs)):
			# abrir img em ./dr
			img = Image.open(path_dref +  ref_imgs[i])
			# Copiar img para ./dc com novo nome 
			new_file_name =  str(nome) + ".jpg"
			img.save(path_dc + new_file_name)
			# Adicionar os novos casos treino à matriz principal
			# Uma imagem fica relacionada tanto com a img_ref front + back 
			M_data.append([ref_imgs[i], new_file_name])
			M_data.append([ref_imgs[i+1], new_file_name])

			# abrir img em ./dr
			img = Image.open(path_dref + ref_imgs[i+1])
			# Copiar img para ./dc com novo nome
			new_file_name =  str(nome+1) + ".jpg"
			img.save(path_dc + new_file_name)
			# Adicionar o novo caso treino à matriz
			M_data.append([ref_imgs[i], new_file_name])
			M_data.append([ref_imgs[i+1], new_file_name])

			i += 2
			nome += 2

		#self.save_csv(M_data, "all_images.csv")
		return M_data
	# -------------------------------------------------------------------------------------------
	
	# -------------------------------------------------------------------------------------------
	#  Dado o ficheiro csv com os dados em bruto, troca a string de cada imagem de referência
	# pelo indice numérico que lhe foi atribuido 
	def troca_nomes_ids(self, ref_classes, M_data):
		ref_names	= []
		classes 	= []
		ref_names	= [x[0] for x in ref_classes]
		classes 	= [x[1] for x in ref_classes]

		# Saltar 1º linha com nome das colunas
		for row in range(1, len(M_data)): 
			index = ref_names.index(M_data[row][0])
			M_data[row][0] = classes[index]

		self.save_csv(M_data, "Dados_final.csv")
		return M_data
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	''' Função para atribuir ids entre [1, 1000] às imagens de referência. 
		As imagens estão ordenadas aos PARES, com uma fotografia da frente (F - front) 
	e verso (B - back) de cada capsula. 
		
		É necessário trocar os nomes das figuras por um valor numérico, que 
	identifique a classe daquele medicamento
		
		A RNA irá devolver na camada de output um valor numérico, correspondente à classe. '''
	def create_ids_classes(self, path_dref):
		# Obter o nome de todas as imagens de referência. 
		# Semelhante a um ls -l de unix 	
		files = os.listdir(path_dref)
		# print("Files found at" + str(path_dref) + ": " + str(len(s)))

		# Trocar o nome das classes de referência por um valor entre [0, N_Classes]
		classe = 1; i=0;
		ref_classes	= []
		#ref_classes.append(["Reference Name", "classe"])
		while(i < len(files)-1):
			ref_classes.append([files[i], classe])
			ref_classes.append([files[i+1], classe])
			classe +=1; 
			i += 2;

		# Criar um csv com o mapeamento entre o Ref-name original e o nr classe atribuido
		self.save_csv(ref_classes, "classes_ref-names.csv")
		
		return ref_classes; 
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	# Função para ler um csv 
	def read_csv_file(self, file_name):
		# Abrir ficheiro CSV
		f = open(file_name, encoding='utf8', mode="r")
		reader = csv.reader(f, delimiter=',')

		# agrupar todas as rows numa lista de listas
		M_data = [] 
		for row in reader:
			# row is an array whit 2 columns
			M_data.append(row)
		f.close()
		
		# Array com o nome das colunas -> linha 0 
		column_names =  M_data[0]
		
		# Obter número de linhas e colunas do ficheiro
		ncols = len(column_names)
		nrows = len(M_data)

		return(M_data)
	# -------------------------------------------------------------------------------------------

	# -------------------------------------------------------------------------------------------
	# Função para guardar um csv 
	def save_csv(self, M_data, file_name):
		with open(file_name, "w+", newline='', encoding='utf-8') as f:
			writer = csv.writer(f, delimiter=",")
			falhou = 0
			for row in range(0,len(M_data)):
					try:        
						writer.writerow(M_data[row])
					except Exception as e:      
						falhou += 1
						f.close()
			print("Save " + file_name + " terminado. Falhou " + str(falhou) + " linhas.")
	
	# -------------------------------------------------------------------------------------------
	def progress(self, count, total, status=''):
		bar_len = 50
		filled_len = int(round(bar_len * count / float(total)))

		percents = round(100.0 * count / float(total), 1)
		bar = '=' * filled_len + '-' * (bar_len - filled_len)

		sys.stdout.write('[%s] %s%s    %s\r' % (bar, percents, '%', status))
		sys.stdout.flush() 
	# -------------------------------------------------------------------------------------------
	

	def Data_Augmentation(self, path_ref, path_dc, M_data):
		start_time = time.time()

		files = os.listdir(path_ref)
		# print("Files found at" + str(path_ref) + ": " + str(len(s)))
		
		pathlib.Path("DataAugmentation").mkdir(parents=True, exist_ok=True) 
		save_to = "./DataAugmentation"

		datagen = ImageDataGenerator(
									rotation_range=40,
									width_shift_range=0.2,
									height_shift_range=0.2,
									rescale=1./255,
									shear_range=0.2,
									zoom_range=0.2,
									horizontal_flip=True,
									fill_mode='nearest')
		i = 0
		classe = 1
		while(i < len(files)):
			text = "Doing 6x Data augmentation of " + str(len(files)) + " images."
			self.progress(i, len(files), text) 

			# Load primeira imagem - Front 
			img = load_img(path_ref + str(files[i])) # Keras function
			img = img_to_array(img) # Keras function
			img = img.reshape((1,) + img.shape) # Keras function
			count = 0			
			for batch in datagen.flow(img, batch_size = 1, save_to_dir=save_to, 
								save_prefix = str(classe), save_format = 'jpeg'):
				count += 1
				if(count > 5):	break
					
			# Load segunda imagem - Back 
			img = load_img(path_ref + str(files[i+1])) # Keras function
			img = img_to_array(img) # Keras function
			img = img.reshape((1,) + img.shape) # Keras function
			count = 0
			for batch in datagen.flow(img, batch_size = 1, save_to_dir=save_to, 
								save_prefix = str(classe), save_format = 'jpeg'):
				count += 1
				if(count > 5):	break

			classe += 1
			i+=2; 
			#print(type(batch))
			#plt.imshow(array_to_img(batch[0])) 
			#plt.show()
			
			# Create 20 new images per reference image 
		
		print("\n")
		print("Tempo Data Augmentation: " + str( round((time.time() - start_time)/60, 2)) + " minutes")
