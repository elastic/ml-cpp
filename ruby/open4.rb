# This class provides extended popen3-style functionality
# by keeping track of running processes, supporting argument
# lists rather than shell command lines, and allowing
# +stdin+, +stdout+ and +stderr+ to be configured separately.
#
# = Example
#
#   require 'open4'
#
#   child = Popen4.new("echo hello; echo there >&2; cat; exit 4")
#   p child.stdout.readline           #=> "hello\n"
#   p child.stderr.readline           #=> "there\n" 
#   child.stdin.puts "Read by cat"
#   p child.stdout.readline           #=> "Read by cat\n"
#   child.stdin.close
#   p child.wait.exitstatus           #=> 4
#
# Beware of the deadlocks that can arise when interacting with
# processes in this manner. If you allow +stderr+ to be made into
# a pipe (the default here) and do not read from it, a process
# will block once it has written a pipe full of data (typically 4k).
# In this case, specify <code>:stdout=>false</code> in the constructor.
#
# To get started, look at the documentation for Popen4.new.
#
#--
# $Id: /local/dcs/trunk/prog/ruby/open4.rb 645 2005-03-15T15:24:02.589341Z jp  $

require 'thread'

class Popen4
  @@active = []

  # call-seq:
  #    Popen4.new(command [, arg, ...] [,options = {}])
  #
  # Starts a process in the background with the same semantics
  # as <code>Kernel::exec</code>. In summary, if a single argument is given,
  # the string is passed to the shell, otherwise the second and subsequent 
  # arguments are passed as parameters to _command_ with no shell 
  # expansion.
  #
  #   Popen4.new("echo *").stdout.read      #=> "file1 file2\n"
  #   Popen4.new("echo","*").stdout.read    #=> "*\n"
  #   Popen4.new("exit 2;").wait.exitstatus #=> 2
  #
  # Keyword options may given to specify the redirection of 
  # +stdin+, +stdout+ and +stderr+. The objects given should be capable
  # of being an argument to IO::reopen, +nil+ to specify
  # no redirection, or a symbol listed below.
  #
  # <code>:stdout</code> or <code>:stderr</code> may be specified to make the
  # file descriptor in question be a copy of an already-allocated
  # pipe. <code>:null</code> indicates redirection to <code>/dev/null</code>.
  #
  # For example, 
  #
  #   child = Popen4.new("command",:stdin=>:null,:stderr=>:stdout)
  #
  # will connect +stdin+ to <code>/dev/null</code> and make both +stdout+ and
  # +stderr+ of the child process available on the <code>child.stdout</code> 
  # stream.
  #
  #   child = Popen4.new("ssh","somewhere",:stderr=>false)
  #
  # The above would be used to make the child process +stdin+ and
  # +stdout+ available to the Ruby script via pipes, but leave +stderr+ 
  # where it was (a terminal, for example).
  #
  # This last example will copy the data between two
  # file objects (they must be backed by a real file descriptor).
  #
  #   Popen4.new("cat",:stdin=>file1,:stdout=>file2).wait
  #
  # Yields self if given a block and closes the pipes before returning.

  def initialize(*cmd)
    Popen4::cleanup

    if Hash === cmd.last then
      options = cmd.pop
    else
      options = {}
    end
    
    @wait_mutex = Mutex.new

    @stdin = nil
    @stdout = nil
    @stderr = nil
    @status = nil
    # List of file handles to close in the parent
    to_close = []

    if options.has_key?(:stdin) then
      child_stdin = options[:stdin]
      if child_stdin == :null then
        child_stdin = File.open("/dev/null","r")
        to_close << child_stdin
      end
    else
      child_stdin, @stdin = IO::pipe
      to_close << child_stdin
    end

    if options.has_key?(:stdout) then
      child_stdout = options[:stdout]
    else
      @stdout, child_stdout = IO::pipe
      to_close << child_stdout
    end

    if options.has_key?(:stderr) then
      child_stderr = options[:stderr]
    else
      @stderr, child_stderr = IO::pipe
      to_close << child_stderr
    end

    # Handle redirections to /dev/null
    if child_stdout == :null then
      child_stdout = File.open("/dev/null","w") 
      to_close << child_stdout
    end
    if child_stderr == :null then
      child_stderr = File.open("/dev/null","w")
      to_close << child_stderr
    end

    # Handle redirections from stdout<->stderr
    child_stdout = child_stderr if child_stdout == :stderr
    child_stderr = child_stdout if child_stderr == :stdout

    @pid = fork do
      if child_stdin then
        @stdin.close if @stdin
        STDIN.reopen(child_stdin)
        child_stdin.close
      end

      if child_stdout then
        @stdout.close if @stdout
        STDOUT.reopen(child_stdout)
      end

      if child_stderr then
        @stderr.close if @stderr
        STDERR.reopen(child_stderr)
        child_stderr.close
      end
      
      if child_stdout
        child_stdout.close unless child_stdout.closed?
      end
      
      begin
        Kernel::exec(*cmd)
      ensure 
        exit!(1)
      end
    end

    to_close.each { |fd| fd.close }
    @stdin.sync = true if @stdin

    Thread.exclusive { @@active << self }
    if block_given? then
      begin
        yield self
      ensure
        close
      end
    end
  end

  # Reap any child processes (by calling poll) as necessary
  def self.cleanup
    active = Thread.exclusive { @@active.dup }
    active.each do |inst|
      inst.poll
    end
  end

  # File handle for pipes to the slave process.
  # Will be +nil+ if a corresponding alternative file
  # was given in the constructor.
  attr_reader :stdin, :stdout, :stderr

  # Process ID of the child
  attr_reader :pid

  # Close the stdin/stdout/stderr pipes from the child process
  # (if they were created in the constructor).
  def close
    [@stdin, @stdout, @stderr].each do |fp|
      begin
        fp.close if fp and not fp.closed?
      rescue
      end
    end
  end
  
  # Wait for the exit status of the process, returning
  # a <code>Process::Status</code> object. The _flags_
  # argument is interpreted as in <code>Process::wait</code>.
  #
  # NB: This wait only returns once the process has actually
  # exited. It does not return for stopped (signaled) processes.
  def wait(flags=0)
    @wait_mutex.synchronize do
      wait_no_lock(flags)
    end
  end
  
  def wait_no_lock(flags=0) #:nodoc:
    return @status if @status

    while result = Process::waitpid2(@pid, flags)
      # Only return exit status
      if result[0] == @pid and (result[1].exited? or result[1].signaled?) then
        @status = result[1]
        Thread.exclusive { @@active.delete(self) }
        return @status
      end
    end
    nil
  end

  private :wait_no_lock

  # Test to see if process has exited without blocking. Returns
  # a <code>Process::Status</code> object or +nil+.
  def poll
    if @wait_mutex.try_lock then
      begin
        wait_no_lock(Process::WNOHANG)
      ensure
        @wait_mutex.unlock
      end
    else
      nil
    end
  end

  alias :status :poll

  # Send the given signal to the process
  def kill(signal)
    Process::kill(signal,@pid)
  end

end

if $0 == __FILE__
  require 'test/unit'

  class TC_Open4 < Test::Unit::TestCase

    def test_default
      p = Popen4.new('read X;echo hello; echo "X was $X"; echo there>&2; exit 4')
      p.stdin.puts "asdf"
      assert_equal "hello",p.stdout.readline.chomp
      assert_equal "X was asdf",p.stdout.readline.chomp
      assert_equal "there",p.stderr.readline.chomp
      assert_equal 4,p.wait.exitstatus
      p.close
    end

    def test_no_stderr
      p = Popen4.new('echo hello; echo ignore this message >&2',:stderr=>false)
      assert_equal "hello",p.stdout.readline.chomp
      assert_equal nil,p.stderr
      assert_equal 0,p.wait.exitstatus
      p.close
    end
    
    def test_no_shell
      Popen4.new('echo','$PATH') do |p|
        # This should be literal '$PATH' and not expanded by the shell
        assert_equal "$PATH", p.stdout.readline.chomp
      end
    end
    
    def test_threaded
      threads = (0..5).map do |idx|
        Thread.new do
          3.times do
            Popen4.new("echo A#{idx};sleep 1;echo B#{idx}",:stderr=>false) do |p|
              assert_equal "A#{idx}", p.stdout.readline.chomp
              assert_equal "B#{idx}", p.stdout.readline.chomp
              assert_equal 0, p.wait.exitstatus  
            end
          end
        end
      end
      threads.each { |t| t.join }  
    end

    def test_kill
      p = Popen4.new <<-EOT
          trap "echo BYE_stderr>&2;echo BYE_stdout" EXIT
          echo start
          sleep 10
          echo notreached
      EOT
      
      assert_equal "start",p.stdout.readline.chomp
      sleep 1
      p.kill('TERM')
      assert_equal "BYE_stdout",p.stdout.readline.chomp
      assert_equal "BYE_stderr",p.stderr.readline.chomp
      assert_equal nil,p.wait.exitstatus
      assert_equal Signal.list['TERM'],p.wait.termsig
    end
  end

end

