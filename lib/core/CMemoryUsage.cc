/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CMemoryUsage.h>

#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CMemoryUsageJsonWriter.h>

#include <sstream>

namespace ml {
namespace core {

namespace memory_detail {

//! Comparison function class to compare CMemoryUsage objects by
//! their description
class CMemoryUsageComparison {
public:
    explicit CMemoryUsageComparison(const std::string& baseline)
        : m_Baseline(baseline) {}

    bool operator()(const CMemoryUsage::TMemoryUsagePtr& rhs) {
        return m_Baseline == rhs->m_Description.s_Name;
    }

private:
    std::string m_Baseline;
};

//! Comparison function class to compare CMemoryUsage objects by
//! their description, but ignoring the first in the collection
class CMemoryUsageComparisonTwo {
public:
    explicit CMemoryUsageComparisonTwo(const std::string& baseline,
                                       const CMemoryUsage::TMemoryUsagePtr& firstItem)
        : m_Baseline(baseline), m_FirstItem(firstItem.get()) {}

    bool operator()(const CMemoryUsage::TMemoryUsagePtr& rhs) {
        return (rhs.get() != m_FirstItem) && (m_Baseline == rhs->m_Description.s_Name);
    }

private:
    std::string m_Baseline;
    const CMemoryUsage* m_FirstItem;
};
}

CMemoryUsage::CMemoryUsage() : m_Description("", 0ull) {
}

CMemoryUsage::TMemoryUsagePtr CMemoryUsage::addChild() {
    auto child = std::make_shared<CMemoryUsage>();
    m_Children.push_back(child);
    return child;
}

CMemoryUsage::TMemoryUsagePtr CMemoryUsage::addChild(std::size_t initialAmount) {
    auto child = std::make_shared<CMemoryUsage>();
    child->m_Description.s_Memory = initialAmount;
    m_Children.push_back(child);
    return child;
}

void CMemoryUsage::addItem(const SMemoryUsage& item) {
    m_Items.push_back(item);
}

void CMemoryUsage::addItem(const std::string& name, std::size_t memory) {
    SMemoryUsage item(name, memory);
    this->addItem(item);
}

void CMemoryUsage::setName(const SMemoryUsage& item) {
    std::size_t initialAmount = m_Description.s_Memory;
    m_Description = item;
    m_Description.s_Memory += initialAmount;
}

void CMemoryUsage::setName(const std::string& name, std::size_t memory) {
    SMemoryUsage item(name, memory);
    this->setName(item);
}

void CMemoryUsage::setName(const std::string& name) {
    SMemoryUsage item(name, 0);
    this->setName(item);
}

std::size_t CMemoryUsage::usage() const {
    std::size_t mem = m_Description.s_Memory;
    for (const auto& item : m_Items) {
        mem += item.s_Memory;
    }

    for (const auto& child : m_Children) {
        mem += child->usage();
    }
    return mem;
}

std::size_t CMemoryUsage::unusage() const {
    std::size_t mem = m_Description.s_Unused;
    for (const auto& item : m_Items) {
        mem += item.s_Unused;
    }

    for (const auto& child : m_Children) {
        mem += child->unusage();
    }
    return mem;
}

void CMemoryUsage::summary(CMemoryUsageJsonWriter& writer) const {
    writer.startObject();
    writer.addItem(m_Description);

    if (m_Items.size() > 0) {
        writer.startArray("items");
        for (const auto& item : m_Items) {
            writer.startObject();
            writer.addItem(item);
            writer.endObject();
        }
        writer.endArray();
    }

    if (!m_Children.empty()) {
        writer.startArray("subItems");
        for (const auto& child : m_Children) {
            child->summary(writer);
        }
        writer.endArray();
    }

    writer.endObject();
}

void CMemoryUsage::compress() {
    using TStrSizeMap = std::map<std::string, std::size_t>;

    if (!m_Children.empty()) {
        // Find out which of the children occur the most
        TStrSizeMap itemsByName;
        for (const auto& child : m_Children) {
            ++itemsByName[child->m_Description.s_Name];
            LOG_TRACE(<< "Item " << child->m_Description.s_Name << " : "
                      << itemsByName[child->m_Description.s_Name]);
        }

        for (const auto& itemByName : itemsByName) {
            // For commonly-occuring children, add up their usage
            // then delete them
            if (itemByName.second > 1) {
                std::size_t counter{0};
                memory_detail::CMemoryUsageComparison compareName{itemByName.first};

                auto firstChildItr = std::find_if(m_Children.begin(),
                                                  m_Children.end(), compareName);
                memory_detail::CMemoryUsageComparisonTwo comparison(
                    itemByName.first, *firstChildItr);

                auto childItr = m_Children.begin();
                while ((childItr = std::find_if(childItr, m_Children.end(), comparison)) !=
                       m_Children.end()) {
                    LOG_TRACE(<< "Trying to remove " << *childItr);
                    (*firstChildItr)->m_Description.s_Memory += (*childItr)->usage();
                    (*firstChildItr)->m_Description.s_Unused += (*childItr)->unusage();
                    childItr = m_Children.erase(childItr);
                    ++counter;
                }
                std::ostringstream ss;
                ss << " [*" << counter + 1 << ']';
                (*firstChildItr)->m_Description.s_Name += ss.str();
            }
        }
    }
    for (auto& child : m_Children) {
        child->compress();
    }
}

void CMemoryUsage::print(std::ostream& outStream) const {
    CMemoryUsageJsonWriter writer(outStream);
    this->summary(writer);
    writer.finalise();
}

} // core
} // ml
