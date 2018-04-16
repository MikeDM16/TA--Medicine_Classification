import csv, os

class Tratamento():

	def tratamento_classes(self, file_name, path_dref):
		M_data = self.read_csv_file(file_name)

		# Obter o nome de todas as imagens de referência		
		s = os.listdir(path_dref)
		# print("Files found at" + str(path_dref) + ": " + str(len(s)))

		''' As imagens estão ordenadas aos PARES, com uma fotografia da frente (F) e
		verso (B) de cada capsula. 
			É necessário trocar os nomes das figuras por um valor numérico, que 
		identifique a classe daquele medicamento '''

		# Trocar o nome das classes de referência por um valor entre [0, N_Classes]
		classe = 1; i=0;
		ref_classes	= []
		ref_classes.append(["Reference Name", "classe"])
		while(i < len(s)-1):
			ref_classes.append([s[i], classe])
			ref_classes.append([s[i+1], classe])
			classe +=1; 
			i += 2;

		# Criar um csv com o mapeamento entre o Ref-name original e o nr classe atribuido
		self.save_csv(ref_classes, "classes_ref-names.csv")

		ref_names	= []
		classes 	= []
		ref_names	= [x[0] for x in ref_classes]
		classes 	= [x[1] for x in ref_classes]

		# Saltar 1º linha com nome das colunas
		for row in range(1, len(M_data)): 
			# substituir o nome de referência pelo id da classe target atribuido 
			index = ref_names.index(M_data[row][0])
			M_data[row][0] = classes[index]

		self.save_csv(M_data, "Limpo.csv")

	def read_csv_file(self, file_name):
		# Abrir ficheiro CSV
		f = open(file_name, encoding='utf8', mode="r")
		reader = csv.reader(f, delimiter=',')

		# agrupar todas as rows numa lista de listas
		M_data = [] # Matrix Nx2 

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