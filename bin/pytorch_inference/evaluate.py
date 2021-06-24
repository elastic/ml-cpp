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
import math
import os
import platform
import stat
import subprocess
import sys

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


def compare_results(expected, actual, tolerance):
    try:
        if expected['request_id'] != actual['request_id']:
            print("request_ids do not match [{}], [{}]".format(expected['request_id'], actual['request_id']), flush=True)
            return False

        if len(expected['inference']) != len(actual['inference']):
            print("len(inference) does not match [{}], [{}]".format(len(expected['inference']), len(actual['inference'])), flush=True)
            return False

        for i in range(len(expected['inference'])):
            expected_row = expected['inference'][i]
            actual_row = actual['inference'][i]

            if len(expected_row) != len(actual_row):
                print("row [{}] lengths are not equal [{}], [{}]".format(i, len(expected_row), len(actual_row)), flush=True)
                return False

            are_close = True
            for j in range(len(expected_row)):
                are_close = are_close and math.isclose(expected_row[j], actual_row[j], abs_tol=tolerance)

            if are_close == False:
                print("row [{}] values are not close {}, {}".format(i, expected_row, actual_row), flush=True)
                return False
    except KeyError as e:
        print("ERROR: comparing results {}. Actual = {}".format(e, actual))
        return False

    return True



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

        print()
        print("reading results...", flush=True)
        with open(args.output_file) as output_file:

            total_time_ms = 0
            doc_count = 0
            results_match = True
            try:
                result_docs = json.load(output_file)
            except:
                print("Error parsing json: ", sys.exc_info()[0])
                return

            for result in result_docs:
                expected = test_evaluation[doc_count]['expected_output']
                 
                tolerance = 1e-04
                if 'how_close' in test_evaluation[doc_count]:
                    tolerance = test_evaluation[doc_count]['how_close']                    


                total_time_ms += result['time']


                # compare to expected
                if compare_results(expected, result, tolerance) == False:
                    print()
                    print('ERROR: inference result [{}] does not match expected results'.format(doc_count))
                    print()
                    results_match = False

                doc_count = doc_count +1

            print()
            print('{} requests evaluated in {} ms'.format(doc_count, total_time_ms))
            print()

            if doc_count != len(test_evaluation): 
                print()
                print('ERROR: The number of inference results [{}] does not match expected count [{}]'.format(doc_count, len(test_evaluation)))
                print()
                results_match = False

            if results_match:
                print()
                print('SUCCESS: inference results match expected', flush=True)
                print()

    finally:
        if os.path.isfile(args.restore_file):
            os.remove(args.restore_file)
        if os.path.isfile(args.input_file):
            os.remove(args.input_file)
        if os.path.isfile(args.output_file):
            os.remove(args.output_file)



if __name__ == "__main__":
    main()

