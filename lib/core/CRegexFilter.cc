/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CRegexFilter.h>

#include <core/CLogger.h>

namespace ml
{
namespace core
{

CRegexFilter::CRegexFilter()
    : m_Regex()
{
}

bool CRegexFilter::configure(const TStrVec &regularExpressions)
{
    m_Regex.clear();
    m_Regex.resize(regularExpressions.size());
    for (std::size_t i = 0; i < regularExpressions.size(); ++i)
    {
        if (m_Regex[i].init(regularExpressions[i]) == false)
        {
            m_Regex.clear();
            LOG_ERROR("Configuration failed; no filtering will apply");
            return false;
        }
    }

    return true;
}

std::string CRegexFilter::apply(const std::string &target) const
{
    if (m_Regex.empty())
    {
        return target;
    }

    std::string result(target);
    std::size_t position = 0;
    std::size_t length = 0;
    for (std::size_t i = 0; i < m_Regex.size(); ++i)
    {
        const CRegex &currentRegex = m_Regex[i];
        while (currentRegex.search(result, position, length))
        {
            result.erase(position, length);
        }
    }
    return result;
}

bool CRegexFilter::empty() const
{
    return m_Regex.empty();
}

}
}
