#!/usr/bin/env python3
import Tratamento, ANN_Keras

def main(): 
	file_name = "groundTruthTable.csv"
	path_dc = "./dc/"
	path_dref = "./dr/"

	T = Tratamento.Tratamento()	
	M_data = T.tratamento_classes(file_name, path_dref, path_dc)
	
	T = ANN_Keras.ANN_Keras(path_dref, path_dc, M_data)

if __name__ == "__main__": main()