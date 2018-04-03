/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CMemory.h>
#include <core/CMemoryUsageJsonWriter.h>
#include <core/CLogger.h>

#include <sstream>

namespace ml
{

namespace core
{

namespace memory_detail
{

//! Comparison function class to compare CMemoryUsage objects by
//! their description
class CMemoryUsageComparison : public std::unary_function<std::string, bool>
{
public:
    explicit CMemoryUsageComparison(const std::string &baseline) : m_Baseline(baseline)
    { }

    bool operator() (const CMemoryUsage *rhs)
    {
        return m_Baseline == rhs->m_Description.s_Name;
    }

private:
    std::string m_Baseline;
};

//! Comparison function class to compare CMemoryUsage objects by
//! their description, but ignoring the first in the collection
class CMemoryUsageComparisonTwo : public std::binary_function<std::string,
                                                              CMemoryUsage *,
                                                              bool>
{
public:
    explicit CMemoryUsageComparisonTwo(const std::string &baseline,
        const CMemoryUsage * firstItem) : m_Baseline(baseline),
        m_FirstItem(firstItem)
    { }

    bool operator() (const CMemoryUsage *rhs)
    {
        return (rhs != m_FirstItem) && (m_Baseline == rhs->m_Description.s_Name);
    }

private:
    std::string m_Baseline;
    const CMemoryUsage * m_FirstItem;
};

}

CMemoryUsage::CMemoryUsage() : m_Description("", 0ull)
{
}

CMemoryUsage::~CMemoryUsage()
{
    for (TMemoryUsagePtrListItr i = m_Children.begin(); i != m_Children.end(); ++i)
    {
        delete *i;
    }
}

CMemoryUsage::TMemoryUsagePtr CMemoryUsage::addChild()
{
    TMemoryUsagePtr child(new CMemoryUsage);
    m_Children.push_back(child);
    return child;
}

CMemoryUsage::TMemoryUsagePtr CMemoryUsage::addChild(std::size_t initialAmount)
{
    TMemoryUsagePtr child(new CMemoryUsage);
    child->m_Description.s_Memory = initialAmount;
    m_Children.push_back(child);
    return child;
}

void CMemoryUsage::addItem(const SMemoryUsage &item)
{
    m_Items.push_back(item);
}

void CMemoryUsage::addItem(const std::string &name, std::size_t memory)
{
    SMemoryUsage item(name, memory);
    this->addItem(item);
}

void CMemoryUsage::setName(const SMemoryUsage &item)
{
    std::size_t initialAmount = m_Description.s_Memory;
    m_Description = item;
    m_Description.s_Memory += initialAmount;
}

void CMemoryUsage::setName(const std::string &name, std::size_t memory)
{
    SMemoryUsage item(name, memory);
    this->setName(item);
}

void CMemoryUsage::setName(const std::string &name)
{
    SMemoryUsage item(name, 0);
    this->setName(item);
}

std::size_t CMemoryUsage::usage(void) const
{
    std::size_t mem = m_Description.s_Memory;
    for (TMemoryUsageVecCitr i = m_Items.begin(); i != m_Items.end(); ++i)
    {
        mem += i->s_Memory;
    }

    for (TMemoryUsagePtrListCItr i = m_Children.begin(); i != m_Children.end(); ++i)
    {
        mem += (*i)->usage();
    }
    return mem;
}

std::size_t CMemoryUsage::unusage(void) const
{
    std::size_t mem = m_Description.s_Unused;
    for (TMemoryUsageVecCitr i = m_Items.begin(); i != m_Items.end(); ++i)
    {
        mem += i->s_Unused;
    }

    for (TMemoryUsagePtrListCItr i = m_Children.begin(); i != m_Children.end(); ++i)
    {
        mem += (*i)->unusage();
    }
    return mem;
}

void CMemoryUsage::summary(CMemoryUsageJsonWriter &writer) const
{
    writer.startObject();
    writer.addItem(m_Description);

    if (m_Items.size() > 0)
    {
        writer.startArray("items");
        for (TMemoryUsageVecCitr i = m_Items.begin(); i != m_Items.end(); ++i)
        {
            writer.startObject();
            writer.addItem(*i);
            writer.endObject();
        }
        writer.endArray();
    }

    if (!m_Children.empty())
    {
        writer.startArray("subItems");
        for (TMemoryUsagePtrListCItr i = m_Children.begin(); i != m_Children.end(); ++i)
        {
            (*i)->summary(writer);
        }
        writer.endArray();
    }

    writer.endObject();
}

void CMemoryUsage::compress(void)
{
    using TStrSizeMap = std::map<std::string, std::size_t>;
    using TStrSizeMapCItr = TStrSizeMap::const_iterator;

    if (!m_Children.empty())
    {
        // Find out which of the children occur the most
        TStrSizeMap itemsByName;
        for (TMemoryUsagePtrListCItr i = m_Children.begin(); i != m_Children.end(); ++i)
        {
            itemsByName[(*i)->m_Description.s_Name]++;
            LOG_TRACE("Item " <<  (*i)->m_Description.s_Name << " : " << itemsByName[(*i)->m_Description.s_Name]);
        }

        for (TStrSizeMapCItr i = itemsByName.begin();
             i != itemsByName.end(); ++i)
        {
            // For commonly-occuring children, add up their usage
            // then delete them
            if (i->second > 1)
            {
                std::size_t counter = 0;
                memory_detail::CMemoryUsageComparison compareName(i->first);

                TMemoryUsagePtrListItr firstChild = std::find_if(m_Children.begin(),
                    m_Children.end(), compareName);
                memory_detail::CMemoryUsageComparisonTwo comparison(i->first, *firstChild);

                TMemoryUsagePtrListItr j = m_Children.begin();
                while ((j = std::find_if(j, m_Children.end(), comparison)) !=
                    m_Children.end())
                {
                    LOG_TRACE("Trying to remove " << *j);
                    (*firstChild)->m_Description.s_Memory += (*j)->usage();
                    (*firstChild)->m_Description.s_Unused += (*j)->unusage();
                    delete *j;
                    j = m_Children.erase(j);
                    counter++;
                }
                std::ostringstream ss;
                ss << " [*" << counter + 1 << "]";
                (*firstChild)->m_Description.s_Name += ss.str();
            }
        }
    }
    for (TMemoryUsagePtrListItr i = m_Children.begin(); i != m_Children.end(); ++i)
    {
        (*i)->compress();
    }
}

void CMemoryUsage::print(std::ostream &outStream) const
{
    CMemoryUsageJsonWriter writer(outStream);
    this->summary(writer);
    writer.finalise();
}

} // core

} // ml
