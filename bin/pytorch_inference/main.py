import argparse
import json
import os
import platform
import random
import stat
import subprocess
import time


#
# python3 signal9.py '/Users/davidkyle/Development/NLP Models/elser_2/elser_model_2.pt' --num_allocations=4
#

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='A TorchScript model with .pt extension')
    parser.add_argument('exec_path', default='pytorch_inference', help='he path to the pytorch_inference executable')
    parser.add_argument('input_file', default='pytorch_requests.txt', help='The path to the file containing the requests')
    # The pipes are created by the C++ app
    # The names must match those passed to app
    parser.add_argument('--output_file', default='out.json')
    parser.add_argument('--log_file', default='log.txt')
    parser.add_argument('--num_threads_per_allocation', type=int, help='The number of inference threads used by LibTorch. Defaults to 1.')
    parser.add_argument('--num_allocations', type=int, help='The number of allocations for parallel forwarding. Defaults to 1')
    parser.add_argument('--cache_size', type=int, help='Cache size limit. Defaults to 0')
    parser.add_argument('--valgrind_log_file', default='valgrind_out.txt', help='Valgrind output file')

    return parser.parse_args()


# def path_to_app2():
#
#     os_platform = platform.system()
#     if os_platform == 'Darwin':
#         if platform.machine() == 'arm64':
#             sub_path = 'darwin-aarch64/controller.app/Contents/MacOS/'
#         else:
#             sub_path = 'darwin-x86_64/controller.app/Contents/MacOS/'
#     elif os_platform == 'Linux':
#         if platform.machine() == 'aarch64':
#             sub_path = 'linux-aarch64/bin/'
#         else:
#             sub_path = 'linux-x86_64/bin/'
#     elif os_platform == 'Windows':
#         sub_path = 'windows-x86_64/bin/'
#     else:
#         raise RuntimeError('Unknown platform')
#
#     return "../../build/distribution/platform/" + sub_path + "pytorch_inference"


def path_to_app(args):
    return args.exec_path


def lauch_pytorch_app(args, input_pipe, restore_file):
    # command = ['valgrind', '--leak-check=full', '--show-leak-kinds=all', '--track-origins=yes', '--verbose',
    #            '--log-file=' + args.valgrind_log_file,
    #            path_to_app(args),
    command = [path_to_app(args),
               '--restore=' + restore_file,
               '--input=' + input_pipe, '--inputIsPipe',
               '--output=' + args.output_file,
               # '--logPipe=' + args.log_file,
               '--validElasticLicenseKeyConfirmed=true',
               ]

    if args.num_threads_per_allocation:
        command.append('--numThreadsPerAllocation=' + str(args.num_threads_per_allocation))

    if args.num_allocations:
        command.append('--numAllocations=' + str(args.num_allocations))

    cache_size_to_use = 0
    if args.cache_size:
        cache_size_to_use = args.cache_size

    command.append('--cacheMemorylimitBytes=' + str(cache_size_to_use))

    # For the memory benchmark always use the immediate executor
    # if args.memory_benchmark:
    #     command.append('--useImmediateExecutor')    
    subprocess.Popen(command)


def stream_file(source, destination):
    while True:
        piece = source.read(8192)
        if not piece:
            break

        destination.write(piece)


def wait_for_pipe(file_name, num_retries=5):
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
        time.sleep(5)

    return stat.S_ISFIFO(os.stat(file_name).st_mode)

def write_mem_usage_request(request_num, destination):
    json.dump({"request_id": "mem_" + str(request_num), "control": 2}, destination)

def write_random_request(request_id, destination):
    json.dump(build_random_inference_request(request_id=request_id), destination)

def build_random_inference_request(request_id):
    num_tokens = 510
    tokens = [101] # CLS
    for _ in range(num_tokens):
        tokens.append(random.randrange(110, 28000))
    tokens.append(102) # SEP

    arg_1 = [1] * (num_tokens + 2)
    arg_2 = [0] * (num_tokens + 2)
    arg_3 = [i for i in range(num_tokens + 2)]

    request = {
        "request_id": request_id,
        "tokens": [tokens],
        "arg_1": [arg_1],
        "arg_2": [arg_2],
        "arg_3": [arg_3],
    }

    return request


def restore_model(model, restore_file_name):
    # create the restore file
    with open(restore_file_name, 'wb') as restore_file:
        file_stats = os.stat(model)
        file_size = file_stats.st_size

        # 4 byte unsigned int
        b = (file_size).to_bytes(4, 'big')
        restore_file.write(b)

        print("streaming model of size", file_size, flush=True)

        with open(model, 'rb') as source_file:
            stream_file(source_file, restore_file)


def main():
    args = parse_arguments()

    input_pipe_name = "model_input"
    restore_file_name = "model_restore_temp"

    # stream the torchscript model
    restore_model(args.model, restore_file_name=restore_file_name)

    lauch_pytorch_app(args, input_pipe=input_pipe_name, restore_file=restore_file_name)

    if not wait_for_pipe(input_pipe_name):
        print("Error: input pipe [{}] has not been created".format(input_pipe_name))
        return

    input_pipe = open(input_pipe_name, 'w')

    print("writing requests")

    request_num = 0
    with open(args.input_file) as file:
        for line in file:
            request_num = request_num + 1
            if request_num % 300 == 0:
                print("Request number: ", request_num)
            # print(line.rstrip())
            input_pipe.write(line.rstrip().rstrip('\n'))


    # i = 0
    # while True:
    #     if i % 100 == 0:
    #         print("mem")
    #         write_mem_usage_request(str(i), input_pipe)
    #     else:
    #         write_random_request(str(i), input_pipe)
    #
    #     i = i + 1

    input_pipe.close()


if __name__ == "__main__":
    main()
