import argparse
import json
import os
import platform
import stat
import subprocess
import time


'''
Reads a TorchScript model from file and streams it via a pipe
to the C++ app. Once the model is loaded the script sends the 
encoded tokens from the input_tokens files and checks the model's 
response matches the expected. 

This script first lauches the C++ pytorch_inference process then
connects the pipes.

Then run this script with input from one of the example directories

python3 evaluate.py /path/to/conll03_traced_ner.pt examples/ner/input.json  examples/ner/expected_response.json

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
    parser.add_argument('--log_pipe', default='log_pipe')

    return parser.parse_args()


def path_to_app():

	os_platform = platform.system()
	if os_platform == 'Darwin':
		sub_path = 'darwin-x86_64/controller.app/Contents/MacOS/'
	elif os_platform == 'Linux':
		# TODO handle the different path for arm architecture 
		sub_path = 'linux-x86_64/'
	elif os_platform == 'Windows':
		sub_path = 'windows-x86_64/'
	else: 
		raise RuntimeError('Unknown platform')


	return "../../build/distribution/platform/" + sub_path + "pytorch_inference"

def lauch_pytorch_app(args):

	command = [path_to_app(), 
		'--restore=' + args.restore_pipe, '--restoreIsPipe', 
		'--input=' + args.input_pipe, '--inputIsPipe',
		'--output=' + args.output_pipe, '--outputIsPipe',
		'--log=' + args.log_pipe, '--logIsPipe',
		'--namedPipeConnectTimeout=3']		
	
	subprocess.Popen(command)


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


def print_logging(fifo):

	print("reading logs")
	line = fifo.readline()
	while line:
		print(line)
		line = fifo.readline()


def main():

	args = parse_arguments()

	lauch_pytorch_app(args)

	# pipes must be connected in a specfic order.
	if not wait_for_pipe(args.log_pipe):	
		print("Error: logging pipe [{}] has not been created".format(args.log_pipe))
		return

	log_pipe = open(args.log_pipe)

	if not wait_for_pipe(args.input_pipe):	
		print("Error: input pipe [{}] has not been created".format(args.input_pipe))
		return

	input_pipe = open('input_pipe', 'wb')

	if not wait_for_pipe(args.output_pipe):	
		print("Error: output pipe [{}] has not been created".format(args.output_pipe))
		return

	output_pipe = open(args.output_pipe)

	if not wait_for_pipe(args.restore_pipe):
		print("Error: restore pipe [{}] has not been created".format(args.restore_pipe))
		return		

	# stream the torchscript model	 
	with open(args.restore_pipe, 'wb') as restore_pipe:
		file_stats = os.stat(args.model)
		file_size = file_stats.st_size
		
		# 4 byte unsigned int
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

	
	print_logging(log_pipe)	


if __name__ == "__main__":
	main()
	

