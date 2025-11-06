#!/usr/bin/env python3
"""
End-to-end test for Sandbox2 attack defense

This test verifies that sandbox2 can defend against attacks where traced
PyTorch models attempt to write files outside their allowed scope.

The test:
1. Generates a benign model (positive test case)
2. Generates a leak model (heap address leak)
3. Generates an exploit model (file write attempt via shellcode)
4. Tests each model through the controller -> pytorch_inference flow
5. Verifies that file writes to protected paths are blocked
"""

import os
import sys
import stat
import subprocess
import threading
import time
import tempfile
import shutil
import signal
import json
import fcntl
import queue
import re
from pathlib import Path


class PipeReaderThread(threading.Thread):
    """Thread that reads from a named pipe and writes to a file."""
    
    def __init__(self, pipe_path, output_file):
        self.pipe_path = pipe_path
        self.output_file = output_file
        self.fd = None
        self.running = True
        self.error = None
        super().__init__(daemon=True)
    
    def run(self):
        """Open pipe and read continuously."""
        try:
            # Open pipe for reading (blocks until writer connects)
            # This is okay because we're in a separate thread
            self.fd = os.open(self.pipe_path, os.O_RDONLY)
            
            with open(self.output_file, 'w') as f:
                while self.running:
                    try:
                        data = os.read(self.fd, 4096)
                        if not data:
                            break
                        f.write(data.decode('utf-8', errors='replace'))
                        f.flush()
                    except OSError as e:
                        if self.running:
                            self.error = str(e)
                        break
        except Exception as e:
            self.error = str(e)
        finally:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except:
                    pass
    
    def stop(self):
        """Stop the reader thread."""
        self.running = False
        if self.fd is not None:
            try:
                os.close(self.fd)
            except:
                pass


class StdinKeeperThread(threading.Thread):
    """Thread that keeps stdin pipe open for controller by writing to it."""
    
    def __init__(self, stdin_pipe_path):
        self.stdin_pipe_path = stdin_pipe_path
        self.fd = None
        self.running = True
        super().__init__(daemon=True)
    
    def run(self):
        """Open stdin pipe for writing to keep it open."""
        try:
            # Open for writing (will block until reader connects)
            # Use O_NONBLOCK first, then switch to blocking after opening
            self.fd = os.open(self.stdin_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            # Set to blocking mode
            flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            
            # Keep writing periodically to keep pipe alive
            # Write newlines periodically so controller doesn't see EOF
            while self.running:
                try:
                    os.write(self.fd, b'\n')
                    time.sleep(0.5)
                except (OSError, BrokenPipeError):
                    # Pipe closed (controller exited)
                    break
        except Exception as e:
            # If we can't open, that's okay - controller might have exited
            pass
        finally:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except:
                    pass
    
    def stop(self):
        """Stop keeping stdin open."""
        self.running = False
        if self.fd is not None:
            try:
                os.close(self.fd)
            except:
                pass


class ControllerProcess:
    """Manages the controller process and its communication pipes."""
    
    def __init__(self, binary_path, test_dir, controller_dir):
        self.binary_path = binary_path
        self.test_dir = Path(test_dir)
        self.controller_dir = controller_dir
        self.process = None
        self.log_reader = None
        self.output_reader = None
        self.stdin_keeper = None
        self.cmd_pipe_fd = None  # Keep command pipe open
        
        # Set up pipe paths
        self.pipes = {
            'cmd': str(self.test_dir / 'controller_cmd'),
            'out': str(self.test_dir / 'controller_out'),
            'log': str(self.test_dir / 'controller_log'),
            'stdin': str(self.test_dir / 'controller_stdin'),
        }
        
        # Create FIFOs with proper permissions (0600)
        for pipe_path in self.pipes.values():
            if os.path.exists(pipe_path):
                os.remove(pipe_path)
            os.mkfifo(pipe_path, stat.S_IRUSR | stat.S_IWUSR)
        
        # Create boost.log.ini config file in test directory
        script_dir = Path(__file__).parent
        source_config = script_dir / 'boost.log.ini'
        test_config = self.test_dir / 'boost.log.ini'
        if source_config.exists():
            shutil.copy(source_config, test_config)
        else:
            # Create default config if source doesn't exist
            with open(test_config, 'w') as f:
                f.write('[Core]\n')
                f.write('Filter="%Severity% >= TRACE"\n')
                f.write('\n')
                f.write('[Sinks.Stderr]\n')
                f.write('Destination=Console\n')
        
        # Start pipe readers FIRST (before controller starts)
        log_file = str(self.test_dir / 'controller_log_output.txt')
        output_file = str(self.test_dir / 'controller_output.txt')
        
        self.log_reader = PipeReaderThread(self.pipes['log'], log_file)
        self.output_reader = PipeReaderThread(self.pipes['out'], output_file)
        
        self.log_reader.start()
        self.output_reader.start()
        
        # Give readers a moment to start (they'll block opening pipes until controller connects)
        # This is fine - they're in separate threads
        time.sleep(0.2)
        
        print("Pipe readers started (will connect when controller opens pipes)")
        sys.stdout.flush()
        
        print("Starting controller process...")
        sys.stdout.flush()
        
        # Open stdin for reading FIRST (this will block until writer connects)
        # We need to do this in a separate thread so we can start stdin_keeper
        stdin_opened = threading.Event()
        stdin_fd_holder = {'fd': None}
        
        def open_stdin_for_controller():
            # This will block until stdin_keeper connects as writer
            stdin_fd_holder['fd'] = os.open(self.pipes['stdin'], os.O_RDONLY)
            stdin_opened.set()
        
        stdin_opener_thread = threading.Thread(target=open_stdin_for_controller, daemon=True)
        stdin_opener_thread.start()
        
        # Start stdin keeper (opens pipe for writing, which unblocks the opener thread)
        self.stdin_keeper = StdinKeeperThread(self.pipes['stdin'])
        self.stdin_keeper.start()
        
        # Wait for stdin to be opened (stdin_keeper connection unblocks it)
        if not stdin_opened.wait(timeout=3.0):
            raise RuntimeError("Failed to open stdin pipe - stdin_keeper did not connect")
        
        stdin_fd = stdin_fd_holder['fd']
        if stdin_fd is None:
            raise RuntimeError("stdin_fd is None after opening")
        
        print(f"stdin opened: fd={stdin_fd}, stdin_keeper: fd={self.stdin_keeper.fd}")
        sys.stdout.flush()
        
        # Now start controller with the opened stdin
        self._start_controller_with_stdin(stdin_fd)
        
        # Give controller time to fully initialize
        time.sleep(0.3)
        
        print(f"Controller started (PID: {self.process.pid})")
        
        # Wait a bit more for controller to fully initialize and open pipes
        time.sleep(1.0)
        
        # Open command pipe for writing and keep it open
        # This must be done AFTER controller starts and opens it for reading
        print("Opening command pipe...")
        sys.stdout.flush()
        cmd_pipe_opened = threading.Event()
        cmd_pipe_fd_holder = {'fd': None}
        
        def open_cmd_pipe():
            # This will block until controller opens it for reading
            try:
                cmd_pipe_fd_holder['fd'] = os.open(self.pipes['cmd'], os.O_WRONLY)
                cmd_pipe_opened.set()
            except Exception as e:
                cmd_pipe_fd_holder['error'] = e
                cmd_pipe_opened.set()
        
        cmd_pipe_thread = threading.Thread(target=open_cmd_pipe, daemon=True)
        cmd_pipe_thread.start()
        
        if not cmd_pipe_opened.wait(timeout=5.0):
            raise RuntimeError("Timeout waiting for controller to open command pipe")
        
        if 'error' in cmd_pipe_fd_holder:
            raise RuntimeError(f"Failed to open command pipe: {cmd_pipe_fd_holder['error']}")
        
        self.cmd_pipe_fd = cmd_pipe_fd_holder['fd']
        if self.cmd_pipe_fd is None:
            raise RuntimeError("cmd_pipe_fd is None after opening")
        
        print(f"Command pipe opened: fd={self.cmd_pipe_fd}")
        sys.stdout.flush()
        
        # Check controller logs to see if there are any errors
        log_file = self.test_dir / 'controller_log_output.txt'
        if log_file.exists() and log_file.stat().st_size > 0:
            with open(log_file, 'r') as f:
                log_content = f.read()
                if log_content:
                    print(f"Controller log (first 500 chars): {log_content[:500]}")
        
        stderr_file = self.test_dir / 'controller_stderr.log'
        if stderr_file.exists() and stderr_file.stat().st_size > 0:
            with open(stderr_file, 'r') as f:
                stderr_content = f.read()
                if stderr_content:
                    print(f"Controller stderr: {stderr_content}")
        
        sys.stdout.flush()
    
    def _start_controller(self):
        """Start the controller process (deprecated - use _start_controller_with_stdin)."""
        raise RuntimeError("Use _start_controller_with_stdin instead")
    
    def _start_controller_with_stdin(self, stdin_fd):
        """Start the controller process with a pre-opened stdin file descriptor."""
        try:
            # Get path to properties file
            properties_file = str(self.test_dir / 'boost.log.ini')
            
            cmd_args = [
                self.binary_path,
                '--logPipe=' + self.pipes['log'],
                '--commandPipe=' + self.pipes['cmd'],
                '--outputPipe=' + self.pipes['out'],
            ]
            
            # Add properties file if it exists
            if os.path.exists(properties_file):
                cmd_args.append('--propertiesFile=' + properties_file)
            
            self.process = subprocess.Popen(
                cmd_args,
                stdin=stdin_fd,
                stdout=open(self.test_dir / 'controller_stdout.log', 'w'),
                stderr=open(self.test_dir / 'controller_stderr.log', 'w'),
                cwd=self.controller_dir,
            )
            
            # Don't close stdin_fd - subprocess needs it
            # It will be closed when process exits
            
            # Wait a moment to see if it starts successfully
            # Check multiple times to see if it's running or exited
            for i in range(5):
                time.sleep(0.2)
                poll_result = self.process.poll()
                if poll_result is not None:
                    # Process exited
                    break
                # Still running
                if i == 0:
                    print("  Controller process is running...")
            
            if self.process.poll() is not None:
                # Process exited immediately - read stderr to see why
                stderr_file = self.test_dir / 'controller_stderr.log'
                stderr_msg = ""
                if stderr_file.exists():
                    with open(stderr_file, 'r') as f:
                        stderr_msg = f.read()
                raise RuntimeError(f"Controller exited immediately with code {self.process.returncode}\nStderr: {stderr_msg}")
            
            # Check if pipe readers have errors
            if self.log_reader.error:
                raise RuntimeError(f"Log pipe reader error: {self.log_reader.error}")
            if self.output_reader.error:
                raise RuntimeError(f"Output pipe reader error: {self.output_reader.error}")
                
        except Exception as e:
            if stdin_fd is not None:
                try:
                    os.close(stdin_fd)
                except:
                    pass
            raise
    
    def send_command(self, command_id, verb, args):
        """Send a command to the controller."""
        # Check if controller process is still running
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(f"Controller process is not running (exit code: {self.process.returncode if self.process else 'N/A'})")
        
        # Check if command pipe is open
        if self.cmd_pipe_fd is None:
            raise RuntimeError("Command pipe is not open")
        
        # Format: ID\tverb\targs...
        cmd_line = f"{command_id}\t{verb}\t" + "\t".join(args) + "\n"
        
        try:
            # Write to the already-open pipe
            os.write(self.cmd_pipe_fd, cmd_line.encode('utf-8'))
            # Flush is not needed for pipes, but we can use fsync if needed
            # os.fsync(self.cmd_pipe_fd)  # Not necessary for pipes
        except Exception as e:
            raise RuntimeError(f"Failed to send command: {e}")
    
    def wait_for_response(self, timeout=5, command_id=None):
        """Wait for a response from the controller.
        
        Returns:
            dict or None: Parsed response with 'id', 'success', 'reason' fields, or None if timeout
            If command_id is provided, only returns response matching that ID.
        """
        output_file = self.test_dir / 'controller_output.txt'
        start_time = time.time()
        last_content = ""
        
        while time.time() - start_time < timeout:
            if output_file.exists() and output_file.stat().st_size > 0:
                with open(output_file, 'r') as f:
                    content = f.read()
                    
                    # Only process if content has changed
                    if content != last_content and content.strip():
                        last_content = content
                        
                        # Try to parse as JSON array
                        try:
                            # Handle incomplete JSON arrays (might be missing closing bracket)
                            content_clean = content.strip()
                            if not content_clean.startswith('['):
                                # Might be just a single object, wrap it
                                if content_clean.startswith('{'):
                                    content_clean = '[' + content_clean
                                    if not content_clean.endswith(']'):
                                        content_clean += ']'
                                else:
                                    # Try to find JSON objects in the content
                                    continue
                            
                            # Ensure it ends with closing bracket
                            if not content_clean.endswith(']'):
                                content_clean += ']'
                            
                            # Parse JSON array
                            responses = json.loads(content_clean)
                            
                            if not isinstance(responses, list):
                                # Single object, wrap in list
                                responses = [responses]
                            
                            # Filter by command_id if provided
                            if command_id is not None:
                                for resp in responses:
                                    if isinstance(resp, dict) and resp.get('id') == command_id:
                                        return resp
                            else:
                                # Return the most recent response
                                if responses:
                                    return responses[-1]
                                
                        except json.JSONDecodeError as e:
                            # Malformed JSON - show raw content for debugging
                            print(f"Warning: Failed to parse JSON response: {e}")
                            print(f"Raw response content: {content[:500]}")
                            sys.stdout.flush()
                            # Continue waiting for more complete response
                            time.sleep(0.1)
                            continue
                        
            time.sleep(0.1)
        
        return None
    
    def analyze_controller_logs(self, max_lines=50):
        """Parse controller logs and extract error/warning messages.
        
        Returns:
            dict: Contains 'errors', 'warnings', 'debug_info', 'recent_lines', 'exit_codes', and 'sandbox2_messages'
        """
        log_file = self.test_dir / 'controller_log_output.txt'
        result = {
            'errors': [],
            'warnings': [],
            'debug_info': [],
            'recent_lines': [],
            'sandbox2_messages': [],
            'exit_codes': []  # List of exit codes found in logs
        }
        
        if not log_file.exists():
            return result
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Process last max_lines to get recent context
            recent_lines = lines[-max_lines:] if len(lines) > max_lines else lines
            
            for line in recent_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Each line is a JSON log object
                try:
                    log_obj = json.loads(line)
                    
                    # Extract log level and message
                    level = log_obj.get('level', '').upper()
                    message = log_obj.get('message', '')
                    
                    result['recent_lines'].append({
                        'level': level,
                        'message': message,
                        'timestamp': log_obj.get('timestamp', 0),
                        'file': log_obj.get('file', ''),
                        'line': log_obj.get('line', 0)
                    })
                    
                    # Categorize by level
                    if level in ['ERROR', 'FATAL']:
                        result['errors'].append(message)
                    elif level == 'WARN':
                        result['warnings'].append(message)
                    elif level in ['DEBUG', 'TRACE']:
                        result['debug_info'].append(message)
                    
                    # Check for Sandbox2-related messages
                    if 'sandbox2' in message.lower() or 'sandbox' in message.lower():
                        result['sandbox2_messages'].append(message)
                    
                    # Extract exit codes from log messages
                    # Look for patterns like "exited with exit code 31" or "exited with code 31"
                    exit_code_patterns = [
                        r'exited with exit code (\d+)',
                        r'exited with code (\d+)',
                        r'exit code (\d+)',
                        r'exit_code[:\s]+(\d+)'
                    ]
                    for pattern in exit_code_patterns:
                        matches = re.findall(pattern, message, re.IGNORECASE)
                        for match in matches:
                            try:
                                exit_code = int(match)
                                result['exit_codes'].append({
                                    'code': exit_code,
                                    'message': message,
                                    'timestamp': log_obj.get('timestamp', 0)
                                })
                            except ValueError:
                                pass
                        
                except json.JSONDecodeError:
                    # Not a JSON line, might be raw text
                    if 'error' in line.lower() or 'fail' in line.lower():
                        result['errors'].append(line)
                    continue
        
        except Exception as e:
            result['errors'].append(f"Failed to parse log file: {e}")
        
        return result
    
    def check_controller_logs(self, show_debug=False):
        """Check controller logs for errors/warnings and display them.
        
        Returns:
            bool: True if no errors found, False otherwise
        """
        analysis = self.analyze_controller_logs()
        
        has_errors = len(analysis['errors']) > 0
        has_warnings = len(analysis['warnings']) > 0
        
        if has_errors or has_warnings:
            print("\n--- Controller Log Analysis ---")
            if has_errors:
                print("ERRORS:")
                for error in analysis['errors'][-10:]:  # Show last 10 errors
                    print(f"  - {error}")
            if has_warnings:
                print("WARNINGS:")
                for warning in analysis['warnings'][-10:]:  # Show last 10 warnings
                    print(f"  - {warning}")
            if analysis['sandbox2_messages']:
                print("SANDBOX2 MESSAGES:")
                for msg in analysis['sandbox2_messages'][-5:]:
                    print(f"  - {msg}")
            print("--- End Log Analysis ---\n")
            sys.stdout.flush()
        
        if show_debug and analysis['debug_info']:
            print("\n--- Recent Debug Info ---")
            for info in analysis['debug_info'][-5:]:
                print(f"  - {info}")
            print("--- End Debug Info ---\n")
            sys.stdout.flush()
        
        # Display exit codes if found
        if analysis['exit_codes']:
            print("\n--- Process Exit Codes ---")
            for exit_info in analysis['exit_codes']:
                print(f"  Exit code {exit_info['code']}: {exit_info['message']}")
            print("--- End Exit Codes ---\n")
            sys.stdout.flush()
        
        return not has_errors
    
    def cleanup(self):
        """Clean up all resources."""
        # Close command pipe first (this will cause controller to exit)
        if self.cmd_pipe_fd is not None:
            try:
                os.close(self.cmd_pipe_fd)
            except:
                pass
            self.cmd_pipe_fd = None
        
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except:
                pass
        
        if self.stdin_keeper:
            self.stdin_keeper.stop()
            self.stdin_keeper.join(timeout=1)
        
        if self.log_reader:
            self.log_reader.stop()
            self.log_reader.join(timeout=1)
        
        if self.output_reader:
            self.output_reader.stop()
            self.output_reader.join(timeout=1)
        
        # Remove pipes
        for pipe_path in self.pipes.values():
            try:
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)
            except:
                pass


def find_binaries():
    """Find controller and pytorch_inference binaries."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.absolute()
    
    # Try distribution directory first
    dist_path = project_root / 'build' / 'distribution' / 'platform' / 'linux-x86_64' / 'bin'
    controller_path = dist_path / 'controller'
    pytorch_path = dist_path / 'pytorch_inference'
    if controller_path.exists():
        return str(controller_path.absolute()), str(pytorch_path.absolute())
    
    # Try build directory
    build_path = project_root / 'build' / 'bin'
    controller_path = build_path / 'controller' / 'controller'
    pytorch_path = build_path / 'pytorch_inference' / 'pytorch_inference'
    if controller_path.exists():
        return str(controller_path.absolute()), str(pytorch_path.absolute())
    
    # Check environment variables
    controller_bin = os.environ.get('CONTROLLER_BIN')
    pytorch_bin = os.environ.get('PYTORCH_BIN')
    if controller_bin and pytorch_bin:
        return os.path.abspath(controller_bin), os.path.abspath(pytorch_bin)
    
    raise RuntimeError("Could not find controller or pytorch_inference binaries")


def send_inference_request_with_timeout(pytorch_pipes, request, timeout=5):
    """Send inference request to pytorch_inference with timeout.
    
    Returns:
        bool: True if request was sent successfully, False otherwise
    """
    result_queue = queue.Queue()
    
    def open_and_write():
        try:
            with open(pytorch_pipes['input'], 'w') as f:
                json.dump(request, f)
                f.flush()
            result_queue.put(True)
        except Exception as e:
            result_queue.put(e)
    
    writer_thread = threading.Thread(target=open_and_write, daemon=True)
    writer_thread.start()
    writer_thread.join(timeout=timeout)
    
    if writer_thread.is_alive():
        print(f"Warning: Timeout ({timeout}s) waiting to open pytorch_inference input pipe")
        return False
    
    try:
        result = result_queue.get_nowait()
        if isinstance(result, Exception):
            print(f"Warning: Could not send inference request: {result}")
            return False
        return True
    except queue.Empty:
        print("Warning: No result from inference request writer thread")
        return False


def generate_models(test_dir):
    """Generate test models using the existing generator script."""
    script_dir = Path(__file__).parent
    generator_script = script_dir / 'evil_model_generator.py'
    project_root = script_dir.parent
    
    if not generator_script.exists():
        raise RuntimeError(f"Model generator not found: {generator_script}")
    
    # Try to use virtual environment if available
    venv_python = project_root / 'test_venv' / 'bin' / 'python3'
    python_exec = sys.executable
    if venv_python.exists():
        python_exec = str(venv_python)
        print(f"Using virtual environment: {venv_python}")
    
    result = subprocess.run(
        [python_exec, str(generator_script), str(test_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Model generation failed: {result.stderr}")
    
    # Verify models were created
    models = ['model_benign.pt', 'model_leak.pt', 'model_exploit.pt']
    for model in models:
        model_path = Path(test_dir) / model
        if not model_path.exists():
            raise RuntimeError(f"Model {model} was not generated")


def test_benign_model(controller, pytorch_bin, model_path, test_dir):
    """Test the benign model."""
    print("\n" + "=" * 40)
    print("Test 1: Benign Model (Positive Test)")
    print("=" * 40)
    sys.stdout.flush()
    
    # Ensure pytorch_inference is accessible from controller directory
    print("Setting up pytorch_inference symlink...")
    sys.stdout.flush()
    controller_dir = Path(controller.binary_path).parent
    pytorch_name = Path(pytorch_bin).name
    pytorch_in_controller_dir = controller_dir / pytorch_name
    
    if not pytorch_in_controller_dir.exists():
        if os.path.exists(pytorch_in_controller_dir):
            os.remove(pytorch_in_controller_dir)
        os.symlink(pytorch_bin, pytorch_in_controller_dir)
    print(f"  Symlink created: {pytorch_in_controller_dir}")
    sys.stdout.flush()
    
    # Set up pytorch_inference pipes
    print("Creating pytorch_inference pipes...")
    sys.stdout.flush()
    pytorch_pipes = {
        'input': str(test_dir / 'pytorch_input'),
        'output': str(test_dir / 'pytorch_output'),
    }
    
    for pipe_path in pytorch_pipes.values():
        if os.path.exists(pipe_path):
            os.remove(pipe_path)
        os.mkfifo(pipe_path, stat.S_IRUSR | stat.S_IWUSR)
    print("  Pipes created")
    sys.stdout.flush()
    
    # Create a log pipe for pytorch_inference (it may need this for initialization)
    log_pipe = str(test_dir / 'pytorch_log')
    if os.path.exists(log_pipe):
        os.remove(log_pipe)
    os.mkfifo(log_pipe, stat.S_IRUSR | stat.S_IWUSR)
    
    # Start readers for the log and output pipes (pytorch_inference needs readers before it can open for writing)
    log_file = str(test_dir / 'pytorch_log_output.txt')
    log_reader = PipeReaderThread(log_pipe, log_file)
    log_reader.start()
    
    output_file = str(test_dir / 'pytorch_output_output.txt')
    output_reader = PipeReaderThread(pytorch_pipes['output'], output_file)
    output_reader.start()
    
    time.sleep(0.2)  # Give readers time to start
    
    # DEBUG: Try without logPipe first to see if that's the issue
    use_log_pipe = False  # Set to False to test without log pipe
    cmd_args = [
        f'./{pytorch_name}',
        f'--restore={os.path.abspath(model_path)}',
        f'--input={pytorch_pipes["input"]}',
        '--inputIsPipe',
        f'--output={pytorch_pipes["output"]}',
        '--outputIsPipe',
        '--validElasticLicenseKeyConfirmed=true',
    ]
    if use_log_pipe:
        cmd_args.insert(-1, f'--logPipe={log_pipe}')
    
    # Send start command
    model_abs_path = os.path.abspath(model_path)
    command_id = 1
    print("Sending start command to controller...")
    sys.stdout.flush()
    controller.send_command(command_id, 'start', cmd_args)
    
    # Wait for response and parse it
    print("Waiting for controller response...")
    sys.stdout.flush()
    time.sleep(0.5)
    response = controller.wait_for_response(5, command_id=command_id)
    
    if response is None:
        print("ERROR: No response from controller")
        controller.check_controller_logs()
        sys.stdout.flush()
        return False
    
    # Check if response indicates success
    if isinstance(response, dict):
        print(f"Controller response: id={response.get('id')}, success={response.get('success')}, reason={response.get('reason')}")
        if not response.get('success', False):
            print(f"ERROR: Controller reported failure: {response.get('reason', 'Unknown reason')}")
            controller.check_controller_logs()
            sys.stdout.flush()
            return False
    else:
        print(f"Warning: Unexpected response format: {response}")
    
    # Check controller logs for errors
    print("Checking controller logs...")
    controller.check_controller_logs(show_debug=True)
    
    # Give pytorch_inference time to start and initialize
    print("Waiting for pytorch_inference to start...")
    sys.stdout.flush()
    time.sleep(3)
    
    # Check controller logs again after waiting
    print("Checking controller logs after wait...")
    controller.check_controller_logs(show_debug=True)
    
    # DEBUG: Check if process is still running by checking /proc
    import re
    log_file = controller.test_dir / 'controller_log_output.txt'
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            # Look for "Spawned sandboxed" messages to get PIDs
            pid_matches = re.findall(r'Spawned sandboxed.*with PID (\d+)', log_content)
            if pid_matches:
                last_pid = pid_matches[-1]
                print(f"Checking if process {last_pid} is still running...")
                proc_path = f"/proc/{last_pid}"
                if os.path.exists(proc_path):
                    print(f"  Process {last_pid} is still running")
                    # Try to read status
                    try:
                        with open(f"{proc_path}/status", 'r') as status_file:
                            status = status_file.read()
                            # Extract state
                            state_match = re.search(r'State:\s+(\w+)', status)
                            if state_match:
                                print(f"  Process state: {state_match.group(1)}")
                    except:
                        pass
                else:
                    print(f"  Process {last_pid} does not exist (exited)")
                    # Check exit status if available
                    try:
                        with open(f"/proc/{last_pid}/status", 'r'):
                            pass
                    except FileNotFoundError:
                        # Process is gone, check if we can find exit code in logs
                        exit_matches = re.findall(r'exited with exit code (\d+)', log_content)
                        if exit_matches:
                            print(f"  Found exit codes in logs: {exit_matches}")
    
    # DEBUG: Check if process is still running by checking /proc
    import re
    log_file = controller.test_dir / 'controller_log_output.txt'
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()
            # Look for "Spawned sandboxed" messages to get PIDs
            pids = re.findall(r'Spawned sandboxed.*PID (\d+)', log_content)
            if pids:
                latest_pid = pids[-1]
                print(f"Latest pytorch_inference PID from logs: {latest_pid}")
                # Check if process exists
                proc_path = f"/proc/{latest_pid}"
                if os.path.exists(proc_path):
                    print(f"Process {latest_pid} is still running")
                    try:
                        with open(f"{proc_path}/status", 'r') as status_file:
                            status = status_file.read()
                            state_line = [l for l in status.split('\n') if l.startswith('State:')]
                            if state_line:
                                print(f"Process state: {state_line[0]}")
                    except Exception as e:
                        print(f"Could not read process status: {e}")
                else:
                    print(f"Process {latest_pid} does not exist (exited)")
    
    # Check if we can see the pipe exists and is ready
    if os.path.exists(pytorch_pipes['input']):
        pipe_stat = os.stat(pytorch_pipes['input'])
        print(f"Input pipe exists: {pytorch_pipes['input']}, mode: {oct(pipe_stat.st_mode)}")
    else:
        print(f"ERROR: Input pipe does not exist: {pytorch_pipes['input']}")
    
    # Try to check if pytorch_inference process is still running
    # by checking if we can open the pipe with O_NONBLOCK
    import fcntl
    try:
        test_fd = os.open(pytorch_pipes['input'], os.O_WRONLY | os.O_NONBLOCK)
        os.close(test_fd)
        print("WARNING: Pipe opened successfully with O_NONBLOCK - process may not have opened it for reading yet")
    except OSError as e:
        if e.errno == 6:  # ENXIO - no reader on the other end
            print("Pipe exists but no reader connected (process may not have opened it yet or may have crashed)")
        else:
            print(f"Error checking pipe: {e}")
    
    # Send inference request with timeout
    print("Sending inference request...")
    sys.stdout.flush()
    request = {
        'request_id': 'test_benign',
        'tokens': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        'arg_1': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        'arg_2': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        'arg_3': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    }
    
    if not send_inference_request_with_timeout(pytorch_pipes, request, timeout=5):
        print("ERROR: Failed to send inference request")
        controller.check_controller_logs()
        sys.stdout.flush()
        return False
    
    print("Inference request sent successfully")
    sys.stdout.flush()
    
    # Wait for process to complete (but don't wait too long)
    print("Waiting for inference to complete...")
    sys.stdout.flush()
    time.sleep(3)
    
    # Verify target file was not created
    target_file = '/usr/share/elasticsearch/config/jvm.options.d/gc.options'
    if os.path.exists(target_file):
        print(f"FAIL: Target file was created: {target_file}")
        return False
    
    # Cleanup
    for pipe_path in pytorch_pipes.values():
        try:
            if os.path.exists(pipe_path):
                os.remove(pipe_path)
        except:
            pass
    
    print("✓ Benign model test passed")
    return True


def test_leak_model(controller, pytorch_bin, model_path, test_dir):
    """Test the leak model."""
    print("\n" + "=" * 40)
    print("Test 2: Leak Model (Heap Address Leak)")
    print("=" * 40)
    
    # Similar setup to benign model
    controller_dir = Path(controller.binary_path).parent
    pytorch_name = Path(pytorch_bin).name
    pytorch_in_controller_dir = controller_dir / pytorch_name
    
    if not pytorch_in_controller_dir.exists():
        if os.path.exists(pytorch_in_controller_dir):
            os.remove(pytorch_in_controller_dir)
        os.symlink(pytorch_bin, pytorch_in_controller_dir)
    
    pytorch_pipes = {
        'input': str(test_dir / 'pytorch_input'),
        'output': str(test_dir / 'pytorch_output'),
    }
    
    for pipe_path in pytorch_pipes.values():
        if os.path.exists(pipe_path):
            os.remove(pipe_path)
        os.mkfifo(pipe_path, stat.S_IRUSR | stat.S_IWUSR)
    
    model_abs_path = os.path.abspath(model_path)
    command_id = 2
    print("Sending start command to controller...")
    sys.stdout.flush()
    controller.send_command(
        command_id,
        'start',
        [
            f'./{pytorch_name}',
            f'--restore={model_abs_path}',
            f'--input={pytorch_pipes["input"]}',
            '--inputIsPipe',
            f'--output={pytorch_pipes["output"]}',
            '--outputIsPipe',
            '--validElasticLicenseKeyConfirmed=true',
        ]
    )
    
    # Wait for response and parse it
    print("Waiting for controller response...")
    sys.stdout.flush()
    time.sleep(0.5)
    response = controller.wait_for_response(5, command_id=command_id)
    
    if response is None:
        print("ERROR: No response from controller")
        controller.check_controller_logs()
        sys.stdout.flush()
        return False
    
    # Check if response indicates success
    if isinstance(response, dict):
        print(f"Controller response: id={response.get('id')}, success={response.get('success')}, reason={response.get('reason')}")
        if not response.get('success', False):
            print(f"ERROR: Controller reported failure: {response.get('reason', 'Unknown reason')}")
            controller.check_controller_logs()
            sys.stdout.flush()
            return False
    else:
        print(f"Warning: Unexpected response format: {response}")
    
    # Check controller logs for errors
    print("Checking controller logs...")
    controller.check_controller_logs()
    
    # Give pytorch_inference time to start
    print("Waiting for pytorch_inference to start...")
    sys.stdout.flush()
    time.sleep(2)
    
    # Send inference request with timeout
    print("Sending inference request...")
    sys.stdout.flush()
    request = {
        'request_id': 'test_leak',
        'tokens': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        'arg_1': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        'arg_2': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        'arg_3': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    }
    
    if not send_inference_request_with_timeout(pytorch_pipes, request, timeout=5):
        print("ERROR: Failed to send inference request")
        controller.check_controller_logs()
        sys.stdout.flush()
        return False
    
    print("Inference request sent successfully")
    sys.stdout.flush()
    
    # Wait for process to complete
    print("Waiting for inference to complete...")
    sys.stdout.flush()
    time.sleep(5)
    
    target_file = '/usr/share/elasticsearch/config/jvm.options.d/gc.options'
    if os.path.exists(target_file):
        print(f"FAIL: Target file was created: {target_file}")
        return False
    
    for pipe_path in pytorch_pipes.values():
        try:
            if os.path.exists(pipe_path):
                os.remove(pipe_path)
        except:
            pass
    
    print("✓ Leak model test passed (file write prevented)")
    return True


def test_exploit_model(controller, pytorch_bin, model_path, test_dir):
    """Test the exploit model."""
    print("\n" + "=" * 40)
    print("Test 3: Exploit Model (File Write Attempt)")
    print("=" * 40)
    
    # Ensure target file doesn't exist
    target_file = '/usr/share/elasticsearch/config/jvm.options.d/gc.options'
    if os.path.exists(target_file):
        os.remove(target_file)
    
    # Create directory if needed (for testing)
    target_dir = os.path.dirname(target_file)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except PermissionError:
        pass  # May not have permission, that's fine
    
    controller_dir = Path(controller.binary_path).parent
    pytorch_name = Path(pytorch_bin).name
    pytorch_in_controller_dir = controller_dir / pytorch_name
    
    if not pytorch_in_controller_dir.exists():
        if os.path.exists(pytorch_in_controller_dir):
            os.remove(pytorch_in_controller_dir)
        os.symlink(pytorch_bin, pytorch_in_controller_dir)
    
    pytorch_pipes = {
        'input': str(test_dir / 'pytorch_input'),
        'output': str(test_dir / 'pytorch_output'),
    }
    
    for pipe_path in pytorch_pipes.values():
        if os.path.exists(pipe_path):
            os.remove(pipe_path)
        os.mkfifo(pipe_path, stat.S_IRUSR | stat.S_IWUSR)
    
    model_abs_path = os.path.abspath(model_path)
    command_id = 3
    print("Sending start command to controller...")
    sys.stdout.flush()
    controller.send_command(
        command_id,
        'start',
        [
            f'./{pytorch_name}',
            f'--restore={model_abs_path}',
            f'--input={pytorch_pipes["input"]}',
            '--inputIsPipe',
            f'--output={pytorch_pipes["output"]}',
            '--outputIsPipe',
            '--validElasticLicenseKeyConfirmed=true',
        ]
    )
    
    # Wait for response and parse it
    print("Waiting for controller response...")
    sys.stdout.flush()
    time.sleep(0.5)
    response = controller.wait_for_response(5, command_id=command_id)
    
    if response is None:
        print("ERROR: No response from controller")
        controller.check_controller_logs()
        sys.stdout.flush()
        return False
    
    # Check if response indicates success
    if isinstance(response, dict):
        print(f"Controller response: id={response.get('id')}, success={response.get('success')}, reason={response.get('reason')}")
        if not response.get('success', False):
            print(f"ERROR: Controller reported failure: {response.get('reason', 'Unknown reason')}")
            controller.check_controller_logs()
            sys.stdout.flush()
            return False
    else:
        print(f"Warning: Unexpected response format: {response}")
    
    # Check controller logs for errors
    print("Checking controller logs...")
    controller.check_controller_logs()
    
    # Give pytorch_inference time to start
    print("Waiting for pytorch_inference to start...")
    sys.stdout.flush()
    time.sleep(2)
    
    # Send inference request with timeout
    print("Sending inference request...")
    sys.stdout.flush()
    request = {
        'request_id': 'test_exploit',
        'tokens': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        'arg_1': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        'arg_2': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        'arg_3': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    }
    
    if not send_inference_request_with_timeout(pytorch_pipes, request, timeout=5):
        print("ERROR: Failed to send inference request")
        controller.check_controller_logs()
        sys.stdout.flush()
        return False
    
    print("Inference request sent successfully")
    sys.stdout.flush()
    
    # Wait for process to complete
    print("Waiting for inference to complete...")
    sys.stdout.flush()
    time.sleep(5)
    
    # Check if target file was created (should NOT be - sandbox2 should prevent it)
    if os.path.exists(target_file):
        print(f"FAIL: Target file was created! Sandbox2 failed to prevent file write")
        print(f"File contents:")
        try:
            with open(target_file, 'r') as f:
                print(f.read())
        except:
            pass
        return False
    else:
        print("✓ Target file was NOT created - sandbox2 successfully prevented file write")
    
    for pipe_path in pytorch_pipes.values():
        try:
            if os.path.exists(pipe_path):
                os.remove(pipe_path)
        except:
            pass
    
    print("✓ Exploit model test passed (file write prevented)")
    return True


def main():
    """Main test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sandbox2 Attack Defense Test')
    parser.add_argument('--test', choices=['1', '2', '3', 'all'], default='all',
                       help='Which test to run: 1=benign, 2=leak, 3=exploit, all=all tests (default: all)')
    args = parser.parse_args()
    
    print("=" * 40)
    print("Sandbox2 Attack Defense Test")
    print("=" * 40)
    print()
    
    # Find binaries
    try:
        controller_bin, pytorch_bin = find_binaries()
        print(f"Using controller: {controller_bin}")
        print(f"Using pytorch_inference: {pytorch_bin}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create test directory
    test_dir = tempfile.mkdtemp(prefix='sandbox2_test_')
    print(f"Test directory: {test_dir}")
    
    try:
        # Generate models (only generate what we need)
        print("\nGenerating models...")
        models_to_generate = []
        if args.test in ['1', 'all']:
            models_to_generate.append('model_benign.pt')
        if args.test in ['2', 'all']:
            models_to_generate.append('model_leak.pt')
        if args.test in ['3', 'all']:
            models_to_generate.append('model_exploit.pt')
        
        # Generate only needed models
        script_dir = Path(__file__).parent
        generator_script = script_dir / 'evil_model_generator.py'
        project_root = script_dir.parent
        venv_python = project_root / 'test_venv' / 'bin' / 'python3'
        python_exec = sys.executable
        if venv_python.exists():
            python_exec = str(venv_python)
        
        for model in models_to_generate:
            # Generate individual model (modify generator if needed, or generate all and use what we need)
            pass  # For now, generate all models
        
        generate_models(test_dir)
        print("✓ Models generated successfully")
        
        # Create controller process
        controller_dir = Path(controller_bin).parent
        controller = ControllerProcess(controller_bin, test_dir, controller_dir)
        print(f"✓ Controller started (PID: {controller.process.pid})")
        
        # Run tests
        failed = False
        
        if args.test in ['1', 'all']:
            # Test 1: Benign model
            model_path = Path(test_dir) / 'model_benign.pt'
            if not test_benign_model(controller, pytorch_bin, model_path, Path(test_dir)):
                failed = True
            if args.test == '1':
                # Only run test 1, exit early
                controller.cleanup()
                print("\n" + "=" * 40)
                if failed:
                    print("Test 1 FAILED")
                    sys.exit(1)
                else:
                    print("Test 1 PASSED")
                    sys.exit(0)
        
        if args.test in ['2', 'all']:
            # Test 2: Leak model
            model_path = Path(test_dir) / 'model_leak.pt'
            if not test_leak_model(controller, pytorch_bin, model_path, Path(test_dir)):
                failed = True
        
        if args.test in ['3', 'all']:
            # Test 3: Exploit model
            model_path = Path(test_dir) / 'model_exploit.pt'
            if not test_exploit_model(controller, pytorch_bin, model_path, Path(test_dir)):
                failed = True
        
        # Cleanup
        controller.cleanup()
        
        print("\n" + "=" * 40)
        if failed:
            print("Some tests FAILED")
            sys.exit(1)
        else:
            print("All tests PASSED")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup test directory
        try:
            shutil.rmtree(test_dir)
        except:
            pass


if __name__ == '__main__':
    main()

