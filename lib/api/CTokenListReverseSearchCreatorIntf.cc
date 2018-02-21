/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CTokenListReverseSearchCreatorIntf.h>


namespace ml
{
namespace api
{


CTokenListReverseSearchCreatorIntf::CTokenListReverseSearchCreatorIntf(const std::string &fieldName)
    : m_FieldName(fieldName)
{
}

CTokenListReverseSearchCreatorIntf::~CTokenListReverseSearchCreatorIntf(void)
{
}

void CTokenListReverseSearchCreatorIntf::closeStandardSearch(std::string &/*part1*/,
                                                             std::string &/*part2*/) const
{
    // Default is to do nothing
}

const std::string &CTokenListReverseSearchCreatorIntf::fieldName(void) const
{
    return m_FieldName;
}


}
}

