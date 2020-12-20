import argparse
import json
import os
import stat
import time


'''
Reads a TorchScript model from file and streams it via a pipe
to the C++ app. Once the model is loaded the script sends the 
encoded tokens from the input_tokens files and checks the model's 
response matches the expected. 

This script expects the C++ process to create the pipes and 
will wait for them to appear -but not for long- so start the 
C++ app first. 

Invoke the C++ from this directory with the command:
   ../../build/distribution/platform/{PLATFORM}/pytorch_inference --restore=restore_pipe --restoreIsPipe --input=input_pipe --inputIsPipe --output=output_pipe --outputIsPipe


replacing {PLATFORM} with your OS specific path   

'''
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='A TorchScript model with .pt extension')
    parser.add_argument('input_tokens', help='JSON file with an array field "tokens"')
    parser.add_argument('expected_output', help='Expected output. Another JSON file with an array field "tokens"')
    # The pipes are created by the C++ app
	# The names must match those passed to app
    parser.add_argument('--restore_pipe', default='restore_pipe')
    parser.add_argument('--input_pipe', default='input_pipe')
    parser.add_argument('--output_pipe', default='output_pipe')

    return parser.parse_args()

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

def write_tokens(fifo, tokens):

	num_tokens = len(tokens)
	fifo.write(num_tokens.to_bytes(4, 'big'))
	for token in tokens:
		fifo.write(token.to_bytes(4, 'big'))



def main():

	args = parse_arguments()

	# input streams must be connected in a specfic order.
	if not wait_for_pipe(args.input_pipe):	
		print("Error: input pipe [{}] has not been created".format(args.input_pipe))
		return

	input_pipe = open('input_pipe', 'wb')

	if not wait_for_pipe(args.output_pipe):	
		print("Error: output pipe [{}] has not been created".format(args.output_pipe))
		return

	output_pipe = open(args.output_pipe)


	if not wait_for_pipe(args.restore_pipe):
		print("Error: timed out waiting for the restore pipe to be created")
		return		


	# stream the torchscript model	 
	with open(args.restore_pipe, 'wb') as restore_pipe:
		file_stats = os.stat(args.model)
		file_size = file_stats.st_size
		
		# TODO is this a signed int?
		b = (file_size).to_bytes(4, 'big')
		restore_pipe.write(b)

		print("streaming model of size", file_size)

		with open(args.model, 'rb') as source_file:
			stream_file(source_file, restore_pipe)


	with open(args.input_tokens) as token_file:
		input_tokens = json.load(token_file)
	
	print("writing query")
	write_tokens(input_pipe, input_tokens['tokens'])
	# one shot inference
	input_pipe.close()


	print("reading results")
	with open(args.expected_output) as expected_output_file:
		expected = json.load(expected_output_file)

	results = json.load(output_pipe)
	# compare to expected
	if results['inference'] == expected['tokens']:
		print('inference results match expected results')
	else:
		print('ERROR: inference results do not match expected results')
		print(results)


if __name__ == "__main__":
	main()
	

