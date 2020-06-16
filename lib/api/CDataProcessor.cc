/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataProcessor.h>

#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CStringUtils.h>
#include <core/CTimeUtils.h>

namespace ml {
namespace api {

// statics
const std::string CDataProcessor::CONTROL_FIELD_NAME(1, CONTROL_FIELD_NAME_CHAR);

CDataProcessor::CDataProcessor(const std::string& timeFieldName, const std::string& timeFieldFormat)
    : m_TimeFieldName{timeFieldName}, m_TimeFieldFormat{timeFieldFormat} {
}

std::string CDataProcessor::debugPrintRecord(const TStrStrUMap& dataRowFields) {
    if (dataRowFields.empty()) {
        return "<EMPTY RECORD>";
    }

    std::string fieldNames;
    std::string fieldValues;
    std::ostringstream result;

    // We want to print the field names on one line, followed by the field
    // values on the next line

    for (TStrStrUMapCItr rowIter = dataRowFields.begin();
         rowIter != dataRowFields.end(); ++rowIter) {
        if (rowIter != dataRowFields.begin()) {
            fieldNames.push_back(',');
            fieldValues.push_back(',');
        }
        fieldNames.append(rowIter->first);
        fieldValues.append(rowIter->second);
    }

    result << fieldNames << core_t::LINE_ENDING << fieldValues;

    return result.str();
}

core_t::TTime CDataProcessor::parseTime(const TStrStrUMap& dataRowFields) const {
    if (m_TimeFieldName.empty()) {
        // No error message here - it's intentional there's no time
        return -1;
    }
    auto iter = dataRowFields.find(m_TimeFieldName);
    if (iter == dataRowFields.end()) {
        ++core::CProgramCounters::counter(counter_t::E_TSADNumberRecordsNoTimeField);
        LOG_ERROR(<< "Found record with no " << m_TimeFieldName << " field:"
                  << core_t::LINE_ENDING << this->debugPrintRecord(dataRowFields));
        return -1;
    }
    core_t::TTime time{-1};
    if (m_TimeFieldFormat.empty()) {
        if (core::CStringUtils::stringToType(iter->second, time) == false) {
            ++core::CProgramCounters::counter(counter_t::E_TSADNumberTimeFieldConversionErrors);
            LOG_ERROR(<< "Cannot interpret " << m_TimeFieldName
                      << " field in record:" << core_t::LINE_ENDING
                      << this->debugPrintRecord(dataRowFields));
            return -1;
        }
    } else {
        // Use this library function instead of raw strptime() as it works
        // around many operating system specific issues.
        if (core::CTimeUtils::strptime(m_TimeFieldFormat, iter->second, time) == false) {
            ++core::CProgramCounters::counter(counter_t::E_TSADNumberTimeFieldConversionErrors);
            LOG_ERROR(<< "Cannot interpret " << m_TimeFieldName << " field using format "
                      << m_TimeFieldFormat << " in record:" << core_t::LINE_ENDING
                      << this->debugPrintRecord(dataRowFields));
            return -1;
        }
    }
    return time;
}

bool CDataProcessor::periodicPersistStateInBackground() {
    // No-op
    return true;
}

bool CDataProcessor::periodicPersistStateInForeground() {
    // No-op
    return true;
}
}
}
