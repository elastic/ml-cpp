/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CStopWatch.h>

#include <core/CLogger.h>

#include <string.h>


namespace ml
{
namespace core
{


CStopWatch::CStopWatch(bool startRunning)
    : m_IsRunning(false),
      m_Start(0),
      m_AccumulatedTime(0)
{
    if (startRunning)
    {
        this->start();
    }
}

void CStopWatch::start()
{
    if (m_IsRunning)
    {
        LOG_ERROR("Stop watch already running");
        return;
    }

    m_IsRunning = true;
    m_Start = m_MonotonicTime.milliseconds();
}

uint64_t CStopWatch::stop()
{
    if (!m_IsRunning)
    {
        LOG_ERROR("Stop watch not running");
        return m_AccumulatedTime;
    }

    m_AccumulatedTime += this->calcDuration();

    m_IsRunning = false;

    return m_AccumulatedTime;
}

uint64_t CStopWatch::lap()
{
    if (!m_IsRunning)
    {
        LOG_ERROR("Stop watch not running");
        return m_AccumulatedTime;
    }

    return m_AccumulatedTime + this->calcDuration();
}

bool CStopWatch::isRunning() const
{
    return m_IsRunning;
}

void CStopWatch::reset(bool startRunning)
{
    m_AccumulatedTime = 0;
    m_IsRunning = false;

    if (startRunning)
    {
        this->start();
    }
}

uint64_t CStopWatch::calcDuration()
{
    uint64_t current(m_MonotonicTime.milliseconds());
    if (current < m_Start)
    {
        LOG_WARN("Monotonic timer has gone backwards - "
                 "stop watch timings will be inaccurate");
        m_Start = current;
        return 0;
    }

    return current - m_Start;
}


}
}

