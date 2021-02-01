'''
Reads a TorchScript model from file and restores it to the C++
app, together with the encoded tokens from the input_tokens
file.  Then it checks the model's response matches the expected.

This script first prepares the input files, then launches the C++
pytorch_inference program which handles them in batch.

Run this script with input from one of the example directories,
for example:

python3 evaluate.py /path/to/conll03_traced_ner.pt examples/ner/input.json  examples/ner/expected_response.json
'''

import argparse
import json
import os
import platform
import stat
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='A TorchScript model with .pt extension')
    parser.add_argument('input_tokens', help='JSON file with an array field "tokens"')
    parser.add_argument('expected_output', help='Expected output. Another JSON file with an array field "tokens"')
    parser.add_argument('--restore_file', default='restore_file')
    parser.add_argument('--input_file', default='input_file')
    parser.add_argument('--output_file', default='output_file')

    return parser.parse_args()

def path_to_app():

    os_platform = platform.system()
    if os_platform == 'Darwin':
        sub_path = 'darwin-x86_64/controller.app/Contents/MacOS/'
    elif os_platform == 'Linux':
		if platform.machine() == 'aarch64':
			sub_path = 'linux-aarch64/bin/'
		else:
			sub_path = 'linux-x86_64/bin/'
    elif os_platform == 'Windows':
        sub_path = 'windows-x86_64/bin/'
    else:
        raise RuntimeError('Unknown platform')

    return "../../build/distribution/platform/" + sub_path + "pytorch_inference"

def launch_pytorch_app(args):

    command = [path_to_app(),
        '--restore=' + args.restore_file,
        '--input=' + args.input_file,
        '--output=' + args.output_file]

    subprocess.Popen(command).communicate()

def stream_file(source, destination) :
    while True:
        piece = source.read(8192)
        if not piece:
            break

        destination.write(piece)

def write_tokens(destination, tokens):

    num_tokens = len(tokens)
    destination.write(num_tokens.to_bytes(4, 'big'))
    for token in tokens:
        destination.write(token.to_bytes(4, 'big'))

def main():

    args = parse_arguments()

    # create the restore file
    with open(args.restore_file, 'wb') as restore_file:
        file_stats = os.stat(args.model)
        file_size = file_stats.st_size

        # 4 byte unsigned int
        b = (file_size).to_bytes(4, 'big')
        restore_file.write(b)

        print("streaming model of size", file_size, flush=True)

        with open(args.model, 'rb') as source_file:
            stream_file(source_file, restore_file)

    with open(args.input_file, 'wb') as input_file:
        with open(args.input_tokens) as token_file:
            input_tokens = json.load(token_file)
        print("writing query", flush=True)
        write_tokens(input_file, input_tokens['tokens'])

    # one shot inference
    launch_pytorch_app(args)

    print("reading results", flush=True)
    with open(args.expected_output) as expected_output_file:
        expected = json.load(expected_output_file)

    with open(args.output_file) as output_file:
        results = json.load(output_file)

    # compare to expected
    if results['inference'] == expected['tokens']:
        print('inference results match expected results')
    else:
        print('ERROR: inference results do not match expected results')
        print(results)


if __name__ == "__main__":
    main()

