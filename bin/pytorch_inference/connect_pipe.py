import os
import sys




# The pipe is created by the C++ app
input_pipe = 'mypipe'
# source_file_name = 'small.txt'
# file_size = 11
source_file_name = '/Users/davidkyle/source/ml-search/projects/universal/torchscript/dbmdz-ner/conll03_traced_ner.pt'
file_size = 1330816933

def streamFile(source, destination) :
	while True:
		piece = source.read(8192)  
		if not piece:
			break

		destination.write(piece)



def main():

	with open(input_pipe, 'wb') as restore_pipe:
		# TODO is this a signed int?
		b = (file_size).to_bytes(4, 'big')
		restore_pipe.write(b)

		print("streaming...")

		with open(source_file_name, 'rb') as source_file:
			streamFile(source_file, restore_pipe)


if __name__ == "__main__":
	main()
	

