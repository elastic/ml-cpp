/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInputParser.h>


namespace ml
{
namespace api
{


CInputParser::CInputParser(void)
    : m_GotFieldNames(false),
      m_GotData(false)
{
}

CInputParser::~CInputParser(void)
{
}

bool CInputParser::gotFieldNames(void) const
{
    return m_GotFieldNames;
}

bool CInputParser::gotData(void) const
{
    return m_GotData;
}

const CInputParser::TStrVec &CInputParser::fieldNames(void) const
{
    return m_FieldNames;
}

void CInputParser::gotFieldNames(bool gotFieldNames)
{
    m_GotFieldNames = gotFieldNames;
}

void CInputParser::gotData(bool gotData)
{
    m_GotData = gotData;
}

CInputParser::TStrVec &CInputParser::fieldNames(void)
{
    return m_FieldNames;
}


}
}

