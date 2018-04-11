/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <test/CTimeSeriesTestData.h>

#include <core/CLogger.h>
#include <core/CRegex.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

#include <boost/bind.hpp>

#include <fstream>

namespace ml {
namespace test {

const std::string CTimeSeriesTestData::DEFAULT_REGEX(
    "\\s*(\\d+|\\d+\\.\\d+)\\s+([-]*\\d+|\\d+\\.\\d+|\\d+\\.\\d+e-?\\d+)\\s*");
const std::string CTimeSeriesTestData::DEFAULT_BIVALUED_REGEX(
    "\\s*(\\d+|\\d+\\.\\d+)\\s+([-]*\\d+|\\d+\\.\\d+|\\d+\\.\\d+e-?\\d+)\\s+([-"
    "]*\\d+|\\d+\\.\\d+|\\d+\\.\\d+e-?\\d+)\\s*");
const std::string CTimeSeriesTestData::DEFAULT_DATE_FORMAT("");
const std::string
    CTimeSeriesTestData::CSV_UNIX_REGEX("^(\\d+),([-]*[\\d\\.]+)");
const std::string CTimeSeriesTestData::CSV_UNIX_BIVALUED_REGEX(
    "^(\\d+),([-]*[\\d\\.]+),([-]*[\\d\\.]+)");
const std::string CTimeSeriesTestData::CSV_UNIX_DATE_FORMAT("");
const std::string CTimeSeriesTestData::CSV_ISO8601_REGEX(
    "^\"(\\d+-\\d+-\\d+T\\d+:\\d+:\\d+)\\..*\",[\"]*([-]*[\\d\\.]+)[\"]*");
const std::string CTimeSeriesTestData::CSV_ISO8601_BIVALUED_REGEX(
    "^\"(\\d+-\\d+-\\d+T\\d+:\\d+:\\d+)\\..*\",[\"]*([-]*[\\d\\.]+)[\"]*, "
    "[\"]*([-]*[\\d\\.]+)[\"]*");
const std::string
    CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT("%Y-%m-%dT%H:%M:%S");

bool CTimeSeriesTestData::parse(const std::string& fileName,
                                TTimeDoublePrVec& results,
                                const std::string& regex,
                                const std::string& dateFormat) {
    core_t::TTime unused(0);

    return CTimeSeriesTestData::parse(fileName, results, unused, unused, regex, dateFormat);
}

bool CTimeSeriesTestData::parse(const std::string& fileName,
                                TTimeDoublePrVec& results,
                                core_t::TTime& minTime,
                                core_t::TTime& maxTime,
                                const std::string& regex,
                                const std::string& dateFormat) {
    return CTimeSeriesTestData::parse(fileName, regex, dateFormat, results, minTime, maxTime);
}

bool CTimeSeriesTestData::parse(const std::string& fileName,
                                TTimeDoubleVecPrVec& results,
                                const std::string& regex,
                                const std::string& dateFormat) {
    core_t::TTime unused(0);

    return CTimeSeriesTestData::parse(fileName, results, unused, unused, regex, dateFormat);
}

bool CTimeSeriesTestData::parse(const std::string& fileName,
                                TTimeDoubleVecPrVec& results,
                                core_t::TTime& minTime,
                                core_t::TTime& maxTime,
                                const std::string& regex,
                                const std::string& dateFormat) {
    return CTimeSeriesTestData::parse(fileName, regex, dateFormat, results, minTime, maxTime);
}

bool CTimeSeriesTestData::parseCounter(const std::string& fileName, TTimeDoublePrVec& results) {
    if (CTimeSeriesTestData::parse(fileName, results) == false) {
        return false;
    }

    double last(0);
    bool started(false);
    for (auto& result : results) {
        double value = result.second;
        if (started == false) {
            result.second = 0;
            started = true;
        } else {
            result.second = value - last;
            if (result.second < 0) {
                LOG_WARN(<< "Negative value " << value << "<" << last << "@"
                         << result.first << " setting counter to 0 ");
                result.second = 0;
            }
        }
        last = value;
    }

    return true;
}

void CTimeSeriesTestData::transform(const TTimeDoublePrVec& data, TDoubleVec& results) {
    results.clear();
    results.reserve(data.size());
    for (const auto& datum : data) {
        results.push_back(datum.second);
    }
}

void CTimeSeriesTestData::derive(const TTimeDoublePrVec& data, TTimeDoublePrVec& results) {
    results.clear();
    if (data.size() <= 1) {
        return;
    }
    results.reserve(data.size() - 1);

    bool hasStarted(false);
    double lastValue(0.0);

    for (const auto& datum : data) {
        if (hasStarted) {
            double v = datum.second - lastValue;

            results.emplace_back(datum.first, v);
        } else {
            hasStarted = true;
        }

        lastValue = datum.second;
    }
}

bool CTimeSeriesTestData::pad(const TTimeDoublePrVec& data,
                              core_t::TTime minTime,
                              core_t::TTime maxTime,
                              TTimeDoublePrVec& results) {
    results.clear();

    if (minTime > maxTime) {
        LOG_ERROR(<< "Invalid bounds " << minTime << ">" << maxTime);
        return false;
    }

    results.reserve(static_cast<size_t>(maxTime - minTime + 1));

    // Inefficient but easy and safe
    using TTimeDoubleMap = std::map<core_t::TTime, double>;

    TTimeDoubleMap dataMap;

    for (const auto& datum : data) {
        if (dataMap.insert({datum.first, datum.second}).second == false) {
            LOG_ERROR(<< "Duplicate values " << datum.first);
            return false;
        }
    }

    for (core_t::TTime t = minTime; t <= maxTime; ++t) {
        auto itr = dataMap.find(t);
        if (itr == dataMap.end()) {
            results.emplace_back(t, 0);
        } else {
            results.emplace_back(t, itr->second);
        }
    }

    return true;
}

namespace {

void add(double value, double& target) {
    target = value;
}

void add(double value, std::vector<double>& target) {
    target.push_back(value);
}
}

template<typename T>
bool CTimeSeriesTestData::parse(const std::string& fileName,
                                const std::string& regex,
                                const std::string& dateFormat,
                                std::vector<std::pair<core_t::TTime, T>>& results,
                                core_t::TTime& minTime,
                                core_t::TTime& maxTime) {
    // reset data
    results.clear();

    std::string tokenRegexString(regex);
    core::CRegex tokenRegex;
    if (tokenRegex.init(tokenRegexString) == false) {
        LOG_ERROR(<< "Regex error");
        return false;
    }

    std::ifstream inputStrm(fileName);
    if (inputStrm.is_open() == false) {
        LOG_ERROR(<< "Unable to read file " << fileName);
        return false;
    }

    std::string line;
    while (std::getline(inputStrm, line)) {
        if (parseLine(tokenRegex, dateFormat, line, results) == false) {
            return false;
        }
    }

    // Must have more than 1 value
    if (results.empty()) {
        LOG_ERROR(<< "Zero values in file " << fileName);
        return false;
    }

    // This assumes the file was sorted in time order
    minTime = results.front().first;
    maxTime = results.back().first;

    return true;
}

template<typename T>
bool CTimeSeriesTestData::parseLine(const core::CRegex& tokenRegex,
                                    const std::string& dateFormat,
                                    const std::string& line,
                                    std::vector<std::pair<core_t::TTime, T>>& results) {
    if (line.empty() || line.find_first_not_of(core::CStringUtils::WHITESPACE_CHARS) ==
                            std::string::npos) {
        LOG_DEBUG(<< "Ignoring blank line");
        return true;
    }

    core::CRegex::TStrVec tokens;
    if (tokenRegex.tokenise(line, tokens) == false) {
        LOG_ERROR(<< "Regex error '" << tokenRegex.str() << "' " << line);
        return false;
    }

    core_t::TTime time(0);
    if (dateFormat.empty()) {
        if (core::CStringUtils::stringToType(tokens[0], time) == false) {
            LOG_ERROR(<< "Invalid test data '" << line << "'");
            return false;
        }
    } else if (core::CTimeUtils::strptime(dateFormat, tokens[0], time) == false) {
        LOG_ERROR(<< "Invalid test data '" << line << "'");
        return false;
    }

    results.emplace_back(time, T());
    for (std::size_t i = 1u; i < tokens.size(); ++i) {
        double value(0.0);
        if (core::CStringUtils::stringToType(tokens[i], value) == false) {
            LOG_ERROR(<< "Invalid test data '" << line << "'");
            return false;
        }
        add(value, results.back().second);
    }

    return true;
}
}
}
