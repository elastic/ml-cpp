#!/usr/bin/env ruby

require 'json'
require 'base64'
require 'zlib'
require 'stringio'

class Dumper

  def initialize filename
    @filename = filename
  end

  def dump
    expected = nil
    actual = nil
    File.read(@filename).each_line do |line|
      if expected.nil?
        if line.start_with? '- Expected: '
          expected = line[12, line.length]
        end
      else
        if line.start_with? '- Actual  : '
          actual = line[12, line.length]
          go_compare expected, actual
          expected = nil
          actual = nil
        end
      end
    end
  end

  def go_compare a, b
    f = File.open 'expected.txt', 'a'
    f << decode(a)
    f.close
    f = File.open 'actual.txt', 'a'
    f << decode(b)
    f.close
  end

  def decode txt
    json = JSON.parse txt
    gz = Zlib::GzipReader.new(StringIO.new(Base64::decode64(json['compressed'].join)))
    gz.read
  end

end


help_message =  "Decode compressed model state from unittest logs\n" + 
                "Writes to expected.txt and actual.txt\n\n" +
                "Usage: dump_model_state.rb <logfile>\n\n"

if ARGV.length == 0 || ARGV.first == '-h'
  STDERR.puts help_message
  exit 1
end

begin
  require 'ap'
rescue
  STDERR.puts "Gem 'awesome_print' not found. Try 'sudo gem install awesome_print'"
  exit 2
end


d = Dumper.new ARGV.first
d.dump


