#!/usr/bin/env python3
import Tratamento

def main(): 
	file_name = "groundTruthTable.csv"
	
	T = Tratamento.Tratamento()	
	T.tratamento_classes(file_name, "dr")
	

if __name__ == "__main__": main()