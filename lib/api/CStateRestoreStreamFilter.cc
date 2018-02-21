/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CStateRestoreStreamFilter.h>

#include <string>

namespace ml
{
namespace api
{

CStateRestoreStreamFilter::CStateRestoreStreamFilter()
    :boost::iostreams::basic_line_filter<char>(true), m_DocCount(0), m_RewrotePreviousLine(false)
{
}

CStateRestoreStreamFilter::string_type CStateRestoreStreamFilter::do_filter(const string_type &line)
{
    // Persist format is:
    // { bulk metadata }
    // { document source }
    // '\0'
    //
    // Restore format is:
    // { Elasticsearch get response }
    // '\0'

    if (line.empty())
    {
        return line;
    }

    size_t leftOffset = 0;
    size_t rightOffset = line.length() - 1;

    if (line[0] == '\0')
    {
        if (line.length() == 1)
        {
            return std::string();
        }
        leftOffset++;
    }

    if (line.compare(leftOffset, 16, "{\"index\":{\"_id\":") == 0)
    {
        m_DocCount++;
        // Strip the leading {"index": and the two closing braces
        leftOffset +=9;

        for (size_t count = 0; count < 2; ++count)
        {
            size_t lastBrace(line.find_last_of('}', rightOffset));

            if (lastBrace != std::string::npos)
            {
                rightOffset = lastBrace - 1;
            }
        }

        m_RewrotePreviousLine = true;

        return line.substr(leftOffset, rightOffset - leftOffset + 1)
                + ",\"_version\":1,\"found\":true,\"_source\":";

    }
    else if (m_RewrotePreviousLine)
    {
        return line + '}' + '\0' + '\n';
    }
    else
    {
        m_RewrotePreviousLine = false;
        return line;
    }
}

size_t CStateRestoreStreamFilter::getDocCount() const
{
    return m_DocCount;
}

}
}

