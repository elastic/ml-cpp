#!/usr/bin/env python3
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

'''
Sustained-load RSS probe for a standalone pytorch_inference process.

Drives a configurable, high-volume stream of inference requests through a
restored TorchScript model (e.g. ELSER) with NO Elasticsearch in the loop,
while sampling the process resident set size (RSS) over time. The goal is a
fast, local reproduction of the nightly QA out-of-memory kill (issue #3226)
without the ~6h ES build + reindex cycle.

The request format and model-restore protocol mirror bin/pytorch_inference/
evaluate.py (the proven standalone harness). Memory is observed two ways:

  * /proc/<pid>/status VmRSS / VmHWM sampled continuously by a watcher thread
    (the authoritative, OS-level view, and the only one that survives an
    OOM-kill); and
  * the process's own "get memory usage" control message (control=2), whose
    `memory_rss` / `memory_max_rss` stats are parsed from the output at the end
    (best effort -- the output is truncated if the process is OOM-killed).

Input can be delivered either as a plain file (default, simplest, bounded by
disk) or as a named pipe (--input-mode pipe, unbounded, streamed on the fly --
this is how Elasticsearch actually drives the process).

The script never fails on an OOM-kill: reproducing the kill IS the success
condition. It prints a clear verdict and the peak RSS, and writes a CSV of the
RSS time series for later plotting.

NOTE: the synthetic tokens are BERT-style, so this is only meaningful for
BERT-based models (ELSER, E5, etc.).

EXAMPLE
-------
    python3 pytorch_inference_rss_probe.py \\
        --app /path/to/platform/linux-x86_64/bin/pytorch_inference \\
        --model /path/to/elser_model_2_linux-x86_64.pt \\
        --num-requests 20000 --batch-size 16 \\
        --num-threads-per-allocation 4 --num-allocations 1 \\
        --csv rss_probe.csv
'''

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

# A short, real WordPiece token sequence (same seed as evaluate.py's memory
# benchmark); padded out to the requested sequence length.
_SEED_TOKENS = [101, 1735, 3912, 18136, 7986, 170, 1647, 109, 126, 119, 122, 3775, 1113, 9031, 102]
_SEED_MASK = [1] * len(_SEED_TOKENS)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--app', required=True,
                        help='Path to the pytorch_inference executable')
    parser.add_argument('--model', required=True,
                        help='TorchScript model (.pt) to restore')
    parser.add_argument('--num-requests', type=int, default=20000,
                        help='Number of inference requests to send (default 20000)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Documents per inference request (default 16)')
    parser.add_argument('--num-tokens', type=int, default=512,
                        help='Padded sequence length per document (default 512)')
    parser.add_argument('--num-threads-per-allocation', type=int, default=None,
                        help='LibTorch intra-op threads (pytorch_inference default if unset)')
    parser.add_argument('--num-allocations', type=int, default=None,
                        help='Parallel forwarding allocations (default 1 if unset)')
    parser.add_argument('--mem-every', type=int, default=100,
                        help='Inject a get-memory control message every N requests (default 100)')
    parser.add_argument('--sample-interval', type=float, default=0.5,
                        help='Seconds between /proc RSS samples (default 0.5)')
    parser.add_argument('--max-seconds', type=float, default=0.0,
                        help='Hard wall-clock cap; terminate the process after this '
                             'many seconds (0 = run to completion, the default)')
    parser.add_argument('--input-mode', choices=['file', 'pipe'], default='file',
                        help='Deliver requests via a plain file (default) or a named pipe')
    parser.add_argument('--work-dir', default='.',
                        help='Directory for the restore/input/output scratch files')
    parser.add_argument('--csv', default='rss_probe.csv',
                        help='Path to write the RSS time-series CSV')
    parser.add_argument('--label', default='',
                        help='Free-form label recorded in the CSV / summary')
    parser.add_argument('--immediate-executor', action='store_true',
                        help='Process requests sequentially (useful for ordered memory readings)')
    return parser.parse_args()


def restore_model(model_path, restore_path):
    '''Write the restore file: a 4-byte big-endian size prefix then the model.'''
    size = os.stat(model_path).st_size
    with open(restore_path, 'wb') as restore_file:
        restore_file.write(size.to_bytes(4, 'big'))
        with open(model_path, 'rb') as source:
            while True:
                chunk = source.read(1 << 20)
                if not chunk:
                    break
                restore_file.write(chunk)
    print('restored model of size {} bytes'.format(size), flush=True)


def make_inference_request(request_num, batch_size, num_tokens):
    tokens = list(_SEED_TOKENS)
    mask = list(_SEED_MASK)
    while len(tokens) < num_tokens:
        tokens.append(0)
        mask.append(0)
    tokens = tokens[:num_tokens]
    mask = mask[:num_tokens]
    zeros = [0] * num_tokens
    ones = [1] * num_tokens
    return {
        'request_id': str(request_num),
        'tokens': [tokens for _ in range(batch_size)],
        'arg_1': [mask for _ in range(batch_size)],
        'arg_2': [zeros for _ in range(batch_size)],
        'arg_3': [ones for _ in range(batch_size)],
    }


def make_mem_request(request_num):
    return {'request_id': 'mem_{}'.format(request_num), 'control': 2}


def write_requests(sink, num_requests, batch_size, num_tokens, mem_every):
    '''Serialise all requests (plus interleaved memory probes) to an open stream.'''
    json.dump(make_mem_request(0), sink)
    for i in range(1, num_requests + 1):
        json.dump(make_inference_request(i, batch_size, num_tokens), sink)
        if mem_every and i % mem_every == 0:
            json.dump(make_mem_request(i), sink)
    json.dump(make_mem_request(num_requests + 1), sink)
    sink.flush()


def build_command(args, restore_path, input_path, output_path):
    command = [
        args.app,
        '--restore=' + restore_path,
        '--input=' + input_path,
        '--output=' + output_path,
        '--validElasticLicenseKeyConfirmed=true',
    ]
    if args.input_mode == 'pipe':
        command.append('--inputIsPipe')
    if args.num_threads_per_allocation is not None:
        command.append('--numThreadsPerAllocation=' + str(args.num_threads_per_allocation))
    if args.num_allocations is not None:
        command.append('--numAllocations=' + str(args.num_allocations))
    if args.immediate_executor:
        command.append('--useImmediateExecutor')
    return command


def read_proc_rss_kb(pid):
    '''Return (VmRSS_kB, VmHWM_kB) for pid, or (None, None) if unavailable.'''
    try:
        with open('/proc/{}/status'.format(pid)) as status:
            rss = hwm = None
            for line in status:
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1])
                elif line.startswith('VmHWM:'):
                    hwm = int(line.split()[1])
            return rss, hwm
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None, None


class RssWatcher(threading.Thread):
    '''Polls /proc RSS for a pid and records a CSV time series until stopped.'''

    def __init__(self, pid, csv_path, interval, label):
        super().__init__(daemon=True)
        self._pid = pid
        self._csv_path = csv_path
        self._interval = interval
        self._label = label
        self._stop = threading.Event()
        self.peak_rss_kb = 0
        self.peak_hwm_kb = 0

    def stop(self):
        self._stop.set()

    def run(self):
        start = time.monotonic()
        last_log = 0.0
        with open(self._csv_path, 'w') as csv:
            csv.write('label,elapsed_s,vmrss_mb,vmhwm_mb\n')
            while not self._stop.is_set():
                rss, hwm = read_proc_rss_kb(self._pid)
                if rss is not None:
                    elapsed = time.monotonic() - start
                    self.peak_rss_kb = max(self.peak_rss_kb, rss)
                    if hwm is not None:
                        self.peak_hwm_kb = max(self.peak_hwm_kb, hwm)
                    csv.write('{},{:.2f},{:.1f},{:.1f}\n'.format(
                        self._label, elapsed, rss / 1024.0,
                        (hwm or 0) / 1024.0))
                    csv.flush()
                    if elapsed - last_log >= 5.0:
                        print('  [rss] t={:6.1f}s  VmRSS={:8.1f} MiB  VmHWM={:8.1f} MiB'.format(
                            elapsed, rss / 1024.0, (hwm or 0) / 1024.0), flush=True)
                        last_log = elapsed
                self._stop.wait(self._interval)


def parse_model_memory(output_path):
    '''Best-effort: return peak model-reported memory_max_rss (bytes) from output.'''
    try:
        with open(output_path) as output_file:
            docs = json.load(output_file)
    except Exception:
        return None
    peak = None
    for doc in docs if isinstance(docs, list) else []:
        stats = doc.get('stats') if isinstance(doc, dict) else None
        if not stats:
            continue
        value = stats.get('memory_max_rss') or stats.get('memory_rss')
        if value is not None:
            peak = value if peak is None else max(peak, value)
    return peak


def run_file_mode(args, command, input_path):
    with open(input_path, 'w') as input_file:
        print('writing {} requests (batch {} x {} tokens) to {}'.format(
            args.num_requests, args.batch_size, args.num_tokens, input_path), flush=True)
        write_requests(input_file, args.num_requests, args.batch_size,
                       args.num_tokens, args.mem_every)
    size_mb = os.stat(input_path).st_size / (1024.0 * 1024.0)
    print('input file is {:.1f} MiB'.format(size_mb), flush=True)
    return subprocess.Popen(command)


def run_pipe_mode(args, command, input_path):
    if os.path.exists(input_path):
        os.remove(input_path)
    os.mkfifo(input_path)
    proc = subprocess.Popen(command)
    # Opening the write end blocks until the process has opened the read end.
    writer = open(input_path, 'w')

    def stream():
        try:
            write_requests(writer, args.num_requests, args.batch_size,
                           args.num_tokens, args.mem_every)
        except BrokenPipeError:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    threading.Thread(target=stream, daemon=True).start()
    return proc


def main():
    args = parse_arguments()
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    restore_path = os.path.join(work_dir, 'restore_file')
    input_path = os.path.join(work_dir, 'input_file')
    output_path = os.path.join(work_dir, 'output_file')

    if platform.system() != 'Linux':
        print('WARNING: /proc RSS sampling only works on Linux; '
              'VmRSS columns will be empty on {}'.format(platform.system()),
              file=sys.stderr, flush=True)

    restore_model(args.model, restore_path)
    command = build_command(args, restore_path, input_path, output_path)
    print('launching: {}'.format(' '.join(command)), flush=True)

    start = time.monotonic()
    try:
        if args.input_mode == 'pipe':
            proc = run_pipe_mode(args, command, input_path)
        else:
            proc = run_file_mode(args, command, input_path)

        watcher = RssWatcher(proc.pid, args.csv, args.sample_interval, args.label)
        watcher.start()
        capped = False
        deadline = (start + args.max_seconds) if args.max_seconds > 0 else None
        while True:
            try:
                returncode = proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() > deadline:
                    print('reached --max-seconds cap ({:.0f}s); terminating'.format(
                        args.max_seconds), flush=True)
                    proc.terminate()
                    try:
                        returncode = proc.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        returncode = proc.wait()
                    capped = True
                    break
        watcher.stop()
        watcher.join(timeout=5.0)
    finally:
        for path in (restore_path, input_path):
            try:
                os.remove(path)
            except OSError:
                pass

    elapsed = time.monotonic() - start
    model_peak = parse_model_memory(output_path)

    print('', flush=True)
    print('==================== RSS PROBE SUMMARY ====================', flush=True)
    if args.label:
        print('label                : {}'.format(args.label), flush=True)
    print('requests             : {} (batch {} x {} tokens)'.format(
        args.num_requests, args.batch_size, args.num_tokens), flush=True)
    print('threads/alloc        : {}'.format(args.num_threads_per_allocation), flush=True)
    print('allocations          : {}'.format(args.num_allocations), flush=True)
    print('wall time            : {:.1f}s'.format(elapsed), flush=True)
    print('peak VmRSS           : {:.1f} MiB ({:.2f} GiB)'.format(
        watcher.peak_rss_kb / 1024.0, watcher.peak_rss_kb / (1024.0 * 1024.0)), flush=True)
    print('peak VmHWM           : {:.1f} MiB ({:.2f} GiB)'.format(
        watcher.peak_hwm_kb / 1024.0, watcher.peak_hwm_kb / (1024.0 * 1024.0)), flush=True)
    if model_peak is not None:
        print('model max_rss        : {:.1f} MiB ({:.2f} GiB)'.format(
            model_peak / (1024.0 * 1024.0), model_peak / (1024.0 * 1024.0 * 1024.0)), flush=True)
    print('process return code  : {}'.format(returncode), flush=True)

    # Negative return code == killed by signal; -9 (SIGKILL) is the OOM-killer.
    if returncode == -9 and not capped:
        print('VERDICT              : *** OOM REPRODUCED *** (pytorch_inference '
              'killed by SIGKILL / OOM killer)', flush=True)
    elif capped:
        print('VERDICT              : time-capped before completion; see RSS slope '
              '(peak {:.2f} GiB)'.format(watcher.peak_rss_kb / (1024.0 * 1024.0)),
              flush=True)
    elif returncode < 0:
        print('VERDICT              : process killed by signal {} (not SIGKILL)'.format(
            -returncode), flush=True)
    elif returncode != 0:
        print('VERDICT              : process exited non-zero ({}) -- check logs'.format(
            returncode), flush=True)
    else:
        print('VERDICT              : completed without OOM', flush=True)
    print('RSS time series      : {}'.format(args.csv), flush=True)
    print('===========================================================', flush=True)

    # The probe itself succeeds whether or not an OOM occurred.
    return 0


if __name__ == '__main__':
    sys.exit(main())
