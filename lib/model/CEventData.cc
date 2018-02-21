/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CEventData.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

namespace ml
{
namespace model
{

namespace
{

const CEventData::TDouble1VecArray DUMMY_ARRAY = CEventData::TDouble1VecArray();
const std::string DASH("-");

}

CEventData::CEventData(void) :
        m_Time(0),
        m_Pid(),
        m_Cids(),
        m_Values(),
        m_IsExplicitNull(false)
{
}

void CEventData::swap(CEventData &other)
{
    std::swap(m_Time, other.m_Time);
    std::swap(m_Pid, other.m_Pid);
    m_Cids.swap(other.m_Cids);
    m_Values.swap(other.m_Values);
    m_StringValue.swap(other.m_StringValue);
    m_Influences.swap(other.m_Influences);
    std::swap(m_IsExplicitNull, other.m_IsExplicitNull);
}

void CEventData::clear(void)
{
    m_Time = 0;
    m_Pid = boost::none;
    m_Cids.clear();
    m_Values.clear();
    m_StringValue = boost::none;
    m_Influences.clear();
    m_IsExplicitNull = false;
}

void CEventData::time(core_t::TTime time)
{
    m_Time = time;
}

bool CEventData::person(std::size_t pid)
{
    if (!m_Pid)
    {
        m_Pid.reset(pid);
    }
    else if (pid != m_Pid)
    {
        LOG_ERROR("Ignoring subsequent person " << pid
                  << ", current person " << *m_Pid);
        return false;
    }
    return true;
}

void CEventData::addAttribute(TOptionalSize cid)
{
    m_Cids.push_back(cid);
}

void CEventData::addValue(const TDouble1Vec &value)
{
    m_Values.push_back(TOptionalDouble1VecArraySizePr());
    if (!value.empty())
    {
        m_Values.back().reset(TDouble1VecArraySizePr(TDouble1VecArray(), 1));
        m_Values.back()->first.fill(value);
        m_Values.back()->second = 1;
    }
}

void CEventData::stringValue(const std::string &value)
{
    m_StringValue.reset(value);
}

void CEventData::addInfluence(const TOptionalStr &influence)
{
    m_Influences.push_back(influence);
}

void CEventData::addCountStatistic(std::size_t count)
{
    TDouble1VecArraySizePr values;
    values.first.fill(TDouble1Vec(1, 0.0));
    values.second = count;
    m_Values.push_back(values);
}

void CEventData::addStatistics(const TDouble1VecArraySizePr &values)
{
    m_Values.push_back(values);
}

core_t::TTime CEventData::time(void) const
{
    return m_Time;
}

CEventData::TOptionalSize CEventData::personId(void) const
{
    return m_Pid;
}

CEventData::TOptionalSize CEventData::attributeId(void) const
{
    if (m_Cids.size() != 1)
    {
        LOG_ERROR("Call to attribute identifier ambiguous: "
                  << core::CContainerPrinter::print(m_Cids));
        return TOptionalSize();
    }
    return m_Cids[0];
}

const CEventData::TDouble1VecArray &CEventData::values(void) const
{
    if (m_Values.size() != 1)
    {
        LOG_ERROR("Call to value ambiguous: "
                  << core::CContainerPrinter::print(m_Values));
        return DUMMY_ARRAY;
    }
    return m_Values[0] ? m_Values[0]->first : DUMMY_ARRAY;
}

const CEventData::TOptionalStr &CEventData::stringValue(void) const
{
    return m_StringValue;
}

const CEventData::TOptionalStrVec &CEventData::influences(void) const
{
    return m_Influences;
}

CEventData::TOptionalSize CEventData::count(void) const
{
    if (m_Values.size() != 1)
    {
        LOG_ERROR("Call to count ambiguous: "
                  << core::CContainerPrinter::print(m_Values));
        return TOptionalSize();
    }
    return m_Values[0] ? m_Values[0]->second : TOptionalSize();
}

std::string CEventData::print(void) const
{
    return   core::CStringUtils::typeToString(m_Time)
           + ' ' + (m_Pid ? core::CStringUtils::typeToString(*m_Pid) : DASH)
           + ' ' + core::CContainerPrinter::print(m_Cids)
           + ' ' + core::CContainerPrinter::print(m_Values);
}

CEventData::TOptionalSize CEventData::attributeId(std::size_t i) const
{
    return i < m_Cids.size() ? m_Cids[i] : TOptionalSize();
}

const CEventData::TDouble1VecArray &CEventData::values(std::size_t i) const
{
    return i < m_Values.size() && m_Values[i] ? m_Values[i]->first : DUMMY_ARRAY;
}

CEventData::TOptionalSize CEventData::count(std::size_t i) const
{
    return i < m_Values.size() && m_Values[i] ? m_Values[i]->second : TOptionalSize();
}

void CEventData::setExplicitNull(void)
{
    // Set count to 0 to avoid checks of count being unset
    this->addCountStatistic(0);

    m_IsExplicitNull = true;
}

bool CEventData::isExplicitNull(void) const
{
    return m_IsExplicitNull;
}

}
}
