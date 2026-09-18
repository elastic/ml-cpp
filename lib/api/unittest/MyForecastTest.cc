/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CJsonOutputStreamWrapper.h>
#include <core/CLogger.h>
#include <core/CRegex.h>
#include <core/CStringUtils.h>
#include <core/Constants.h>

#include <model/CAnomalyDetectorModelConfig.h>
#include <model/CLimits.h>

#include <api/CAnomalyJobConfig.h>

#include <test/CTimeSeriesTestData.h>

#include "CTestAnomalyJob.h"

#include <boost/json.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(MyForecastTest)

using namespace ml;
using namespace boost;

namespace {
double importCsvUtilizationData(const std::string& fileName, CTestAnomalyJob& job) {
    std::ifstream ifs(fileName.c_str());
    BOOST_TEST_REQUIRE(ifs.is_open());

    ml::core::CRegex regex;
    BOOST_TEST_REQUIRE(regex.init(","));

    std::string line;
    // Skip header line
    BOOST_TEST_REQUIRE(std::getline(ifs, line).good());

    std::size_t recordCount = 0;
    double maxUtilization = 0.0;

    while (std::getline(ifs, line)) {
        LOG_TRACE(<< "Processing line: " << line);
        
        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        ml::core::CRegex::TStrVec tokens;
        regex.split(line, tokens);

        // Skip lines that don't have exactly 3 columns
        if (tokens.size() != 3) {
            LOG_WARN(<< "Skipping line with " << tokens.size() << " tokens: " << line);
            continue;
        }

        // Track maximum utilization for validation
        double utilizationValue = 0.0;
        BOOST_TEST_REQUIRE(ml::core::CStringUtils::stringToType(tokens[2], utilizationValue));
        maxUtilization = std::max(maxUtilization, utilizationValue);

        CTestAnomalyJob::TStrStrUMap dataRow;
        dataRow["time"] = tokens[0];
        dataRow["utilization"] = tokens[2];  // Use the utilization column

        BOOST_TEST_REQUIRE(job.handleRecord(dataRow));
        recordCount++;
    }
    
    LOG_INFO(<< "Processed " << recordCount << " records for testing");
    LOG_INFO(<< "Maximum utilization observed: " << maxUtilization);
    
    return maxUtilization;
}

void validateForecastResults(std::stringstream& outputStrm, double maxInputUtilization, std::size_t expectedForecastCount) {
    std::string output = outputStrm.str();
    LOG_DEBUG(<< "JSON output length: " << output.length());
    
    if (output.empty()) {
        LOG_ERROR(<< "Empty JSON output");
        BOOST_TEST_REQUIRE(false);
        return;
    }

    
    // Basic validation: check that we got forecast output
    BOOST_TEST_REQUIRE(output.find("model_forecast") != std::string::npos);
    
    
    // Additional validation: Check that forecast predictions don't exceed max observed utilization
    double maxForecastPrediction = 0.0;
    std::size_t forecastCount = 0;
    
    // Parse JSON line by line to extract forecast_prediction values
    std::istringstream stream(output);
    std::string line;
    
    while (std::getline(stream, line)) {
        // Look for model_forecast entries
        if (line.find("model_forecast") != std::string::npos && 
            line.find("forecast_prediction") != std::string::npos) {
            
            // Extract forecast_prediction value using regex
            std::size_t predPos = line.find("\"forecast_prediction\":");
            if (predPos != std::string::npos) {
                std::size_t valueStart = line.find(':', predPos) + 1;
                std::size_t valueEnd = line.find_first_of(",}", valueStart);
                
                if (valueStart != std::string::npos && valueEnd != std::string::npos) {
                    std::string valueStr = line.substr(valueStart, valueEnd - valueStart);
                    double forecastValue = 0.0;
                    
                    if (ml::core::CStringUtils::stringToType(valueStr, forecastValue)) {
                        maxForecastPrediction = std::max(maxForecastPrediction, forecastValue);
                        forecastCount++;
                    }
                }
            }
        }
    }
    
    LOG_INFO(<< "Found " << forecastCount << " forecast predictions");
    LOG_INFO(<< "Maximum forecast prediction: " << maxForecastPrediction);
    LOG_INFO(<< "Maximum input utilization: " << maxInputUtilization);
    
    // Assert that forecast predictions don't exceed historical maximum
    // Allow for reasonable tolerance due to trends and model uncertainty
    double tolerance = maxInputUtilization * 0.05; // 5% tolerance for trend extrapolation
    BOOST_TEST_REQUIRE(forecastCount == expectedForecastCount); // Must have the correct number of forecasts
    BOOST_REQUIRE_LE(maxForecastPrediction, maxInputUtilization + tolerance);
    
    LOG_INFO(<< "Forecast validation passed - all predictions within reasonable bounds of historical data");
}
}  // anonymous namespace

BOOST_AUTO_TEST_CASE(testAnomalyDetectionWithForecast) {
    LOG_INFO(<< "Starting testAnomalyDetectionWithForecast");
    // Test configuration
    static const core_t::TTime BUCKET_LENGTH = 900;  // 15 minute buckets (to match data frequency)
    static const core_t::TTime FORECAST_DURATION = 14L * 24L * 3600L;  // 14 days
    const std::string CSV_FILENAME = "testfiles/doc_count_intput.csv";
    static const std::size_t EXPECTED_FORECAST_COUNT = 14L * 24L * 4L; // 14 days * 24 hours * 4 hours per bucket

    std::stringstream outputStrm;
    double maxUtilization = 0.0;
    {
        // Step 1: Create the anomaly detector (scoped to ensure proper cleanup)
        model::CLimits limits;
        api::CAnomalyJobConfig jobConfig = 
            CTestAnomalyJob::makeSimpleJobConfig("mean", "utilization", "", "", "", {});
        
        model::CAnomalyDetectorModelConfig modelConfig =
            model::CAnomalyDetectorModelConfig::defaultConfig(BUCKET_LENGTH);
        
        ml::core::CJsonOutputStreamWrapper streamWrapper(outputStrm);
        
        CTestAnomalyJob job("test_job", limits, jobConfig, modelConfig, streamWrapper);

        // Step 2: Load and process CSV data (and capture maximum utilization)
        maxUtilization = importCsvUtilizationData(CSV_FILENAME, job);

        // Step 3: Trigger 14-day forecast
        CTestAnomalyJob::TStrStrUMap forecastRequest;
        forecastRequest["."] = "p{\"duration\":" + std::to_string(FORECAST_DURATION) + 
                              ",\"forecast_id\": \"test_forecast_14d\"" + 
                              ",\"create_time\": \"" + std::to_string(time(nullptr)) + "\" }";
        
        BOOST_TEST_REQUIRE(job.handleRecord(forecastRequest));
        
        LOG_INFO(<< "Forecast request submitted, waiting for job completion...");
        
        // Job destructor will be called here, which waits for all async operations (forecasts) to complete
    }
    
    // Step 4: Now all forecasts are guaranteed to be complete, validate results
    LOG_INFO(<< "Job completed, validating forecast results");
    validateForecastResults(outputStrm, maxUtilization, EXPECTED_FORECAST_COUNT);
}

BOOST_AUTO_TEST_SUITE_END()
