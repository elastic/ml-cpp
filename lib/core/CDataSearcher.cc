/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CDataSearcher.h>


namespace ml
{
namespace core
{

const std::string CDataSearcher::EMPTY_STRING;


CDataSearcher::CDataSearcher(void)
    : m_SearchTerms(2)
{
}

CDataSearcher::~CDataSearcher(void)
{
}

void CDataSearcher::setStateRestoreSearch(const std::string &index)
{
    m_SearchTerms[0] = index;
    m_SearchTerms[1].clear();
}

void CDataSearcher::setStateRestoreSearch(const std::string &index,
                                          const std::string &id)
{
    m_SearchTerms[0] = index;
    m_SearchTerms[1] = id;
}


}
}
