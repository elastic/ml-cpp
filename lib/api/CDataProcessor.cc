/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataProcessor.h>

#include <core/CLogger.h>

namespace ml
{
namespace api
{

// statics
const std::string CDataProcessor::CONTROL_FIELD_NAME(1, CONTROL_FIELD_NAME_CHAR);

CDataProcessor::CDataProcessor()
{
}

CDataProcessor::~CDataProcessor()
{
    // Most compilers put the vtable in the object file containing the
    // definition of the first non-inlined virtual function, so DON'T move this
    // empty definition to the header file!
}

std::string CDataProcessor::debugPrintRecord(const TStrStrUMap &dataRowFields)
{
    if (dataRowFields.empty())
    {
        return "<EMPTY RECORD>";
    }

    std::string fieldNames;
    std::string fieldValues;
    std::ostringstream result;

    // We want to print the field names on one line, followed by the field
    // values on the next line

    for (TStrStrUMapCItr rowIter = dataRowFields.begin();
         rowIter != dataRowFields.end();
         ++rowIter)
    {
        if (rowIter != dataRowFields.begin())
        {
            fieldNames.push_back(',');
            fieldValues.push_back(',');
        }
        fieldNames.append(rowIter->first);
        fieldValues.append(rowIter->second);
    }

    result << fieldNames << core_t::LINE_ENDING << fieldValues;

    return result.str();
}

bool CDataProcessor::periodicPersistState(CBackgroundPersister &/*persister*/)
{
    // No-op
    return true;
}


}
}

