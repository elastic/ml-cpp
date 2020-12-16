import os
import stat
import time



# The pipes are created by the C++ app
# The names must match those passed to app
restore_pipe_name = 'restore_pipe'
input_pipe_name = 'input_pipe'
output_pipe_name = 'output_pipe'


source_file_name = '/Users/davidkyle/source/ml-search/projects/universal/torchscript/dbmdz-ner/conll03_traced_ner.pt'
file_size = 1330816933

def stream_file(source, destination) :
	while True:
		piece = source.read(8192)  
		if not piece:
			break

		destination.write(piece)

def wait_for_pipe(file_name, num_retries=5) :
	'''
	the pipe must exist else it will be created as an 
	ordinary file when opened for write
	'''
	while num_retries > 0:
		try:
			if stat.S_ISFIFO(os.stat(file_name).st_mode):
				break
		except Exception:
			pass


		num_retries = num_retries -1
		time.sleep(0.05)

	return stat.S_ISFIFO(os.stat(file_name).st_mode)

def write_tokens(fifo):

	# hardcoded query based on the example 
	# https://github.com/elastic/ml-search/tree/master/projects/universal/torchscript/dbmdz-ner
	tokens = [101, 20164, 10932, 10289, 3561, 119, 1110, 170, 1419, 1359, 1107, 1203, 1365, 1392, 119, 2098, 3834, 1132, 1107, 141, 25810, 23904, 117, 3335, 1304, 1665, 20316, 1106, 1103, 6545, 3640, 119, 102]

	for token in tokens:
		fifo.write(token.to_bytes(4, 'big'))



def main():

	# input streams must be connected in a specfic order.

	if not wait_for_pipe(input_pipe_name):	
		print("Error: input pipe [{}] has not been created".format(input_pipe_name))
		return

	input_pipe = open(input_pipe_name, 'wb')

	if not wait_for_pipe(output_pipe_name):	
		print("Error: output pipe [{}] has not been created".format(output_pipe_name))
		return

	output_pipe = open(output_pipe_name)


	if not wait_for_pipe(restore_pipe_name):
		print("Error: timed out waiting for the restore pipe to be created")
		return		


	with open(restore_pipe_name, 'wb') as restore_pipe:
		# TODO is this a signed int?
		b = (file_size).to_bytes(4, 'big')
		restore_pipe.write(b)

		print("streaming model...")

		with open(source_file_name, 'rb') as source_file:
			stream_file(source_file, restore_pipe)


	print("writing query")
	write_tokens(input_pipe)
	# one shot inference
	input_pipe.close()


	print("reading results")
	# results = output_pipe.read()


if __name__ == "__main__":
	main()
	

