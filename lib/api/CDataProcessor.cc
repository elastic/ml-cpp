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
#include <api/CDataProcessor.h>

#include <core/CLogger.h>

namespace ml {
namespace api {

// statics
const std::string CDataProcessor::CONTROL_FIELD_NAME(1, CONTROL_FIELD_NAME_CHAR);

CDataProcessor::CDataProcessor(void)
{}

CDataProcessor::~CDataProcessor(void) {
    // Most compilers put the vtable in the object file containing the
    // definition of the first non-inlined virtual function, so DON'T move this
    // empty definition to the header file!
}

std::string CDataProcessor::debugPrintRecord(const TStrStrUMap &dataRowFields) {
    if (dataRowFields.empty()) {
        return "<EMPTY RECORD>";
    }

    std::string fieldNames;
    std::string fieldValues;
    std::ostringstream result;

    // We want to print the field names on one line, followed by the field
    // values on the next line

    for (TStrStrUMapCItr rowIter = dataRowFields.begin();
         rowIter != dataRowFields.end();
         ++rowIter) {
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

bool CDataProcessor::periodicPersistState(CBackgroundPersister & /*persister*/) {
    // No-op
    return true;
}


}
}

