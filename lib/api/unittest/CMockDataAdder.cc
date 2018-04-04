/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CMockDataAdder.h"

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include "CMockSearcher.h"


CMockDataAdder::CMockDataAdder()
{
}

CMockDataAdder::TOStreamP CMockDataAdder::addStreamed(const std::string &index,
                                                      const std::string &/*id*/)
{
    LOG_TRACE("Add Streamed for index " << index);
    if (m_Streams.find(index) == m_Streams.end())
    {
        m_Streams[index] = TOStreamP(new std::ostringstream);
    }
    return m_Streams[index];
}

bool CMockDataAdder::streamComplete(TOStreamP &strm,
                                    bool /*force*/)
{
    LOG_TRACE("Stream complete");
    bool found = false;
    for (TStrOStreamPMapItr i = m_Streams.begin(); i != m_Streams.end(); ++i)
    {
        if (i->second == strm)
        {
            LOG_TRACE("Found stream for " << i->first);
            std::ostringstream *ss = dynamic_cast<std::ostringstream *>(i->second.get());
            if (ss != 0)
            {
                const std::string &result = ss->str();
                LOG_TRACE("Adding data: " << result);
                m_Events[i->first].push_back('[' + result + ']');
                found = true;
            }
        }
    }
    return found;
}

const CMockDataAdder::TStrStrVecMap &CMockDataAdder::events() const
{
    return m_Events;
}

void CMockDataAdder::clear()
{
    m_Events.clear();
    m_Streams.clear();
}

