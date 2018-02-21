#!/usr/bin/env ruby

require 'optparse'
require 'json'
require 'net/http'
require 'uri'
require_relative 'open4.rb'

class Dumper

  def initialize id, file, options
    @id = id
    @options = options
    @file = file
  end

  def dump
    # Get the snapshots list
    http = Net::HTTP.new @options[:host], @options[:port]
    base_url = "/mlresults-#{@id}"

    query = {}
    query['sort'] = []
    query['sort'] << { 'restorePriority' => { 'order' => 'desc' } }
    request = Net::HTTP::Get.new "#{base_url}/modelSnapshot/_search?size=1"
    request.body = query.to_json
    request['Content-Type'] = 'application/json'
    response = http.request request
    if response.code.to_i != 200
      STDERR.puts "Response from server was #{response.code}: #{response.body}"
      exit 2
    end
    json = JSON.parse response.body
    json = json['hits']['hits'].first['_source']

    num_docs = json['snapshotDocCount'].to_i
    ss_id = json['snapshotId']

    STDERR.puts "Querying: #{num_docs} docs"
    f = File.new(@file, 'wb')

    process = Popen4.new('base64 -D | gzip -d -c', :stdout => f, :stderr => STDERR)

    num_docs.times do |i|
      request = Net::HTTP::Get.new "#{base_url}/modelState/#{ss_id}_#{i+1}"
      response = http.request request
      exit 3 unless response.code.to_i == 200
      state = JSON.parse response.body
      state['_source']['compressed'].each do |str|
        process.stdin.write str
      end
    end
    STDERR.puts "Finished reading docs"
    process.stdin.flush
    process.stdin.close
    process.wait
    STDERR.puts "Finished decoding"
  end

end


help_message =  "Extract the persisted model state for a job and decode it\n" + 
                "Prints the result to STDOUT\n\n" + 
                "Usage: dump_model_state.rb [options] <ID> <output file>\n\n"
options = {}
OptionParser.new do |opts|
  opts.banner = help_message

  options[:host] = 'localhost'
  opts.on("-h", "--host HOSTNAME", "Use this hostname instead of localhost") do |v|
    options[:host] = v
  end

  options[:port] = '9200'
  opts.on("-p", "--port", "Use this port instead of 9200") do |v|
    options[:port] = v
  end

end.parse!(ARGV)

if ARGV.length < 2
  STDERR.puts help_message
  STDERR.puts "Model state ID required and output file required"
  exit 1
end

STDERR.puts "Getting model state for #{ARGV.first}"

d = Dumper.new ARGV[0], ARGV[1], options
d.dump


