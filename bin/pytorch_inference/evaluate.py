'''
Reads a TorchScript model from file and restores it to the C++
app, together with the encoded tokens from the input_tokens
file.  Then it checks the model's response matches the expected.

This script reads the input files and expected outputs, then 
launches the C++ pytorch_inference program which handles and 
sends the request. The response is checked against the expected 
defined in the test file

The test file must have the format:
[
    {
        "input": {"request_id": "foo", "tokens": [1, 2, 3]},
        "expected_output": {"request_id": "foo", "inference": [1, 2, 3]}
    },
    ...
]


EXAMPLES
--------
Run this script with input from one of the example directories,
for example:

python3 evaluate.py /path/to/conll03_traced_ner.pt examples/ner/test_run.json 
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
    parser.add_argument('test_file', help='JSON file with an array of objects each '
     'containing "input" and "expected_output" subobjects')
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

def write_request(request, destination):    
    json.dump(request, destination)

def main():

    args = parse_arguments()

    try:                    
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

        with open(args.input_file, 'w') as input_file:
            with open(args.test_file) as test_file:
                test_evaluation = json.load(test_file)
            print("writing query", flush=True)
            for doc in test_evaluation:
                write_request(doc['input'], input_file)
        
        launch_pytorch_app(args)

        print("reading results", flush=True)    
        with open(args.output_file) as output_file:
            
            doc_count = 0
            results_match = True 
            # output is NDJSON
            for jsonline in output_file:
                result = json.loads(jsonline)       
                expected = test_evaluation[doc_count]['expected_output']

                # compare to expected
                if result != expected:
                    print('ERROR: inference result [{}] does not match expected results'.format(doc_count))
                    print(result, expected)
                    results_match = False

                doc_count = doc_count +1

            if results_match:
                print('SUCCESS: inference results match expected')

    finally:        
        if os.path.isfile(args.restore_file):
            os.remove(args.restore_file)
        if os.path.isfile(args.input_file):
            os.remove(args.input_file)
        if os.path.isfile(args.output_file):
            os.remove(args.output_file)                        



if __name__ == "__main__":
    main()

