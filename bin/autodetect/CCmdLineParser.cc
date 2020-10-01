/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CCmdLineParser.h"

#include <ver/CBuildInfo.h>

#include <model/CAnomalyDetector.h>
#include <model/CAnomalyScore.h>

#include <boost/program_options.hpp>

#include <iostream>

namespace ml {
namespace autodetect {

const std::string CCmdLineParser::DESCRIPTION = "Usage: autodetect [options] [<fieldname>+ [by <fieldname>]]\n"
                                                "Options:";

bool CCmdLineParser::parse(int argc,
                           const char* const* argv,
                           std::string& limitConfigFile,
                           std::string& modelConfigFile,
                           std::string& fieldConfigFile,
                           std::string& modelPlotConfigFile,
                           std::string& jobId,
                           std::string& logProperties,
                           std::string& logPipe,
                           core_t::TTime& bucketSpan,
                           core_t::TTime& latency,
                           std::string& summaryCountFieldName,
                           char& delimiter,
                           bool& lengthEncodedInput,
                           std::string& timeField,
                           std::string& timeFormat,
                           std::string& quantilesState,
                           bool& deleteStateFiles,
                           core_t::TTime& persistInterval,
                           std::size_t& bucketPersistInterval,
                           core_t::TTime& maxQuantileInterval,
                           core_t::TTime& namedPipeConnectTimeout,
                           std::string& inputFileName,
                           bool& isInputFileNamedPipe,
                           std::string& outputFileName,
                           bool& isOutputFileNamedPipe,
                           std::string& restoreFileName,
                           bool& isRestoreFileNamedPipe,
                           std::string& persistFileName,
                           bool& isPersistFileNamedPipe,
                           bool& isPersistInForeground,
                           std::size_t& maxAnomalyRecords,
                           bool& memoryUsage,
                           bool& multivariateByFields,
                           bool& stopCategorizationOnWarnStatus,
                           TStrVec& clauseTokens) {
    try {
        boost::program_options::options_description desc(DESCRIPTION);
        // clang-format off
        desc.add_options()
            ("help", "Display this information and exit")
            ("version", "Display version information and exit")
            ("limitconfig", boost::program_options::value<std::string>(),
                        "Optional limit config file")
            ("modelconfig", boost::program_options::value<std::string>(),
                        "Optional model config file")
            ("fieldconfig", boost::program_options::value<std::string>(),
                        "Optional field config file")
            ("modelplotconfig", boost::program_options::value<std::string>(),
                        "Optional model plot config file")
            ("jobid", boost::program_options::value<std::string>(),
                        "ID of the job this process is associated with")
            ("logProperties", boost::program_options::value<std::string>(),
                        "Optional logger properties file")
            ("logPipe", boost::program_options::value<std::string>(),
                        "Optional log to named pipe")
            ("bucketspan", boost::program_options::value<core_t::TTime>(),
                        "Optional aggregation bucket span (in seconds) - default is 300")
            ("latency", boost::program_options::value<core_t::TTime>(),
                        "Optional maximum delay for out-of-order records (in seconds) - default is 0")
            ("summarycountfield", boost::program_options::value<std::string>(),
                        "Optional field to that contains counts for pre-summarized input - default is none")
            ("delimiter", boost::program_options::value<char>(),
                        "Optional delimiter character for delimited data formats - default is '\t' (tab separated)")
            ("lengthEncodedInput",
                        "Take input in length encoded binary format - default is delimited")
            ("timefield", boost::program_options::value<std::string>(),
                        "Optional name of the field containing the timestamp - default is 'time'")
            ("timeformat", boost::program_options::value<std::string>(),
                        "Optional format of the date in the time field in strptime code - default is the epoch time in seconds")
            ("quantilesState", boost::program_options::value<std::string>(),
                        "Optional file to quantiles for normalization")
            ("deleteStateFiles",
                        "If the 'quantilesState' option is used and this flag is set then delete the model state files once they have been read")
            ("namedPipeConnectTimeout", boost::program_options::value<core_t::TTime>(),
                        "Optional timeout (in seconds) for connecting named pipes on startup - default is 300 seconds")
            ("input", boost::program_options::value<std::string>(),
                        "Optional file to read input from - not present means read from STDIN")
            ("inputIsPipe", "Specified input file is a named pipe")
            ("output", boost::program_options::value<std::string>(),
                        "Optional file to write output to - not present means write to STDOUT")
            ("outputIsPipe", "Specified output file is a named pipe")
            ("restore", boost::program_options::value<std::string>(),
                        "Optional file to restore state from - not present means no state restoration")
            ("restoreIsPipe", "Specified restore file is a named pipe")
            ("persist", boost::program_options::value<std::string>(),
                        "Optional file to persist state to - not present means no state persistence")
            ("persistIsPipe", "Specified persist file is a named pipe")
            ("persistInterval", boost::program_options::value<core_t::TTime>(),
                        "Optional time interval at which to periodically persist model state (Mutually exclusive with bucketPersistInterval)")
            ("persistInForeground", "Persistence occurs in the foreground. Defaults to background persistence.")
            ("bucketPersistInterval", boost::program_options::value<std::size_t>(),
                        "Optional number of buckets after which to periodically persist model state (Mutually exclusive with persistInterval)")
            ("maxQuantileInterval", boost::program_options::value<core_t::TTime>(),
                        "Optional interval at which to periodically output quantiles if they have not been output due to an anomaly - if not specified then quantiles will only be output following a big anomaly")
            ("maxAnomalyRecords", boost::program_options::value<std::size_t>(),
                        "The maximum number of records to be outputted for each bucket. Defaults to 100, a value 0 removes the limit.")
            ("memoryUsage",
                        "Log the model memory usage at the end of the job")
            ("multivariateByFields",
                        "Optional flag to enable multi-variate analysis of correlated by fields")
            ("stopCategorizationOnWarnStatus",
                        "Optional flag to stop categorization for partitions where the status is 'warn'.")
        ;
        // clang-format on

        boost::program_options::variables_map vm;
        boost::program_options::parsed_options parsed =
            boost::program_options::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run();
        boost::program_options::store(parsed, vm);

        auto checkConflictingOptions = [&vm](const std::string& opt1,
                                             const std::string& opt2) {
            if (vm.count(opt1) && (vm[opt1].defaulted() == false) &&
                vm.count(opt2) && (vm[opt2].defaulted() == false))
                throw std::runtime_error("Conflicting options '" + opt1 +
                                         "' and '" + opt2 + "'.");
        };

        if (vm.count("help") > 0) {
            std::cerr << desc << std::endl;
            return false;
        }
        if (vm.count("version") > 0) {
            std::cerr << "Model State Version "
                      << model::CAnomalyDetector::STATE_VERSION << std::endl
                      << "Quantile State Version "
                      << model::CAnomalyScore::CURRENT_FORMAT_VERSION << std::endl
                      << ver::CBuildInfo::fullInfo() << std::endl;
            return false;
        }
        if (vm.count("limitconfig") > 0) {
            limitConfigFile = vm["limitconfig"].as<std::string>();
        }
        if (vm.count("modelconfig") > 0) {
            modelConfigFile = vm["modelconfig"].as<std::string>();
        }
        if (vm.count("fieldconfig") > 0) {
            fieldConfigFile = vm["fieldconfig"].as<std::string>();
        }
        if (vm.count("modelplotconfig") > 0) {
            modelPlotConfigFile = vm["modelplotconfig"].as<std::string>();
        }
        if (vm.count("jobid") > 0) {
            jobId = vm["jobid"].as<std::string>();
        }
        if (vm.count("logProperties") > 0) {
            logProperties = vm["logProperties"].as<std::string>();
        }
        if (vm.count("logPipe") > 0) {
            logPipe = vm["logPipe"].as<std::string>();
        }
        if (vm.count("bucketspan") > 0) {
            bucketSpan = vm["bucketspan"].as<core_t::TTime>();
        }
        if (vm.count("latency") > 0) {
            latency = vm["latency"].as<core_t::TTime>();
        }
        if (vm.count("summarycountfield") > 0) {
            summaryCountFieldName = vm["summarycountfield"].as<std::string>();
        }
        if (vm.count("delimiter") > 0) {
            delimiter = vm["delimiter"].as<char>();
        }
        if (vm.count("lengthEncodedInput") > 0) {
            lengthEncodedInput = true;
        }
        if (vm.count("timefield") > 0) {
            timeField = vm["timefield"].as<std::string>();
        }
        if (vm.count("timeformat") > 0) {
            timeFormat = vm["timeformat"].as<std::string>();
        }
        if (vm.count("quantilesState") > 0) {
            quantilesState = vm["quantilesState"].as<std::string>();
        }
        if (vm.count("deleteStateFiles") > 0) {
            deleteStateFiles = true;
        }
        checkConflictingOptions("persistInterval", "bucketPersistInterval");
        if (vm.count("persistInterval") > 0) {
            persistInterval = vm["persistInterval"].as<core_t::TTime>();
        }
        if (vm.count("bucketPersistInterval") > 0) {
            bucketPersistInterval = vm["bucketPersistInterval"].as<std::size_t>();
        }
        if (vm.count("maxQuantileInterval") > 0) {
            maxQuantileInterval = vm["maxQuantileInterval"].as<core_t::TTime>();
        }
        if (vm.count("namedPipeConnectTimeout") > 0) {
            namedPipeConnectTimeout = vm["namedPipeConnectTimeout"].as<core_t::TTime>();
        }
        if (vm.count("input") > 0) {
            inputFileName = vm["input"].as<std::string>();
        }
        if (vm.count("inputIsPipe") > 0) {
            isInputFileNamedPipe = true;
        }
        if (vm.count("output") > 0) {
            outputFileName = vm["output"].as<std::string>();
        }
        if (vm.count("outputIsPipe") > 0) {
            isOutputFileNamedPipe = true;
        }
        if (vm.count("restore") > 0) {
            restoreFileName = vm["restore"].as<std::string>();
        }
        if (vm.count("restoreIsPipe") > 0) {
            isRestoreFileNamedPipe = true;
        }
        if (vm.count("persist") > 0) {
            persistFileName = vm["persist"].as<std::string>();
        }
        if (vm.count("persistIsPipe") > 0) {
            isPersistFileNamedPipe = true;
        }
        if (vm.count("persistInForeground") > 0) {
            isPersistInForeground = true;
        }
        if (vm.count("maxAnomalyRecords") > 0) {
            maxAnomalyRecords = vm["maxAnomalyRecords"].as<std::size_t>();
        }
        if (vm.count("memoryUsage") > 0) {
            memoryUsage = true;
        }
        if (vm.count("multivariateByFields") > 0) {
            multivariateByFields = true;
        }
        if (vm.count("stopCategorizationOnWarnStatus") > 0) {
            stopCategorizationOnWarnStatus = true;
        }

        boost::program_options::collect_unrecognized(
            parsed.options, boost::program_options::include_positional)
            .swap(clauseTokens);
    } catch (std::exception& e) {
        std::cerr << "Error processing command line: " << e.what() << std::endl;
        return false;
    }

    return true;
}
}
}
