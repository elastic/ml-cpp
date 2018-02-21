/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CDataTyper.h>


namespace ml
{
namespace api
{


// Initialise statics
const CDataTyper::TStrStrUMap CDataTyper::EMPTY_FIELDS;


CDataTyper::CDataTyper(const std::string &fieldName)
    : m_FieldName(fieldName),
      m_LastPersistTime(0)
{
}

CDataTyper::~CDataTyper(void)
{
}

int CDataTyper::computeType(bool isDryRun,
                            const std::string &str,
                            size_t rawStringLen)
{
    return this->computeType(isDryRun, EMPTY_FIELDS, str, rawStringLen);
}

const std::string &CDataTyper::fieldName(void) const
{
    return m_FieldName;
}

core_t::TTime CDataTyper::lastPersistTime(void) const
{
    return m_LastPersistTime;
}

void CDataTyper::lastPersistTime(core_t::TTime lastPersistTime)
{
    m_LastPersistTime = lastPersistTime;
}


}
}

