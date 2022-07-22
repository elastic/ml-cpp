/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */
#include <core/CMemoryUsage.h>

#include <core/CLoggerTrace.h>
#include <core/CMemory.h>
#include <core/CMemoryUsageJsonWriter.h>

#include <list>
#include <map>
#include <sstream>
#include <vector>

namespace ml {
namespace core {

class CMemoryUsage::CImpl {
public:
    static CImpl& impl(TMemoryUsagePtr ptr) { return *ptr->m_Impl; }

public:
    CImpl() : m_Description{"", 0ULL} {}

    TMemoryUsagePtr addChild() {
        auto child = std::make_shared<CMemoryUsage>();
        m_Children.push_back(child);
        return child;
    }

    TMemoryUsagePtr addChild(std::size_t initialAmount) {
        auto child = std::make_shared<CMemoryUsage>();
        impl(child).m_Description.s_Memory = initialAmount;
        m_Children.push_back(child);
        return child;
    }

    void addItem(const SMemoryUsage& item) { m_Items.push_back(item); }

    void addItem(const std::string& name, std::size_t memory) {
        SMemoryUsage item(name, memory);
        this->addItem(item);
    }

    void setName(const SMemoryUsage& item) {
        std::size_t initialAmount = m_Description.s_Memory;
        m_Description = item;
        m_Description.s_Memory += initialAmount;
    }

    void setName(const std::string& name, std::size_t memory) {
        SMemoryUsage item(name, memory);
        this->setName(item);
    }

    void setName(const std::string& name) {
        SMemoryUsage item(name, 0);
        this->setName(item);
    }

    std::size_t usage() const {
        std::size_t mem = m_Description.s_Memory;
        for (const auto& item : m_Items) {
            mem += item.s_Memory;
        }

        for (const auto& child : m_Children) {
            mem += child->usage();
        }
        return mem;
    }

    std::size_t unusage() const {
        std::size_t mem = m_Description.s_Unused;
        for (const auto& item : m_Items) {
            mem += item.s_Unused;
        }

        for (const auto& child : m_Children) {
            mem += child->unusage();
        }
        return mem;
    }

    void print(std::ostream& outStream) const {
        CMemoryUsageJsonWriter writer(outStream);
        this->summary(writer);
        writer.finalise();
    }

    void compress() {
        using TStrSizeMap = std::map<std::string, std::size_t>;

        if (!m_Children.empty()) {
            TStrSizeMap itemsByName;
            for (const auto& child : m_Children) {
                ++itemsByName[impl(child).m_Description.s_Name];
                LOG_TRACE(<< "Item " << child->m_Description.s_Name << " : "
                          << itemsByName[child->m_Description.s_Name]);
            }

            for (const auto & [ name, count ] : itemsByName) {
                // Add up the usage of duplicate items and then delete them.
                if (count > 1) {
                    auto equal = [name_ = name](const TMemoryUsagePtr& child) {
                        return impl(child).m_Description.s_Name == name_;
                    };

                    auto firstChildItr = std::find_if(m_Children.begin(),
                                                      m_Children.end(), equal);

                    auto childItr = firstChildItr;
                    ++childItr;
                    while ((childItr = std::find_if(childItr, m_Children.end(), equal)) !=
                           m_Children.end()) {
                        LOG_TRACE(<< "Trying to remove " << *childItr);
                        impl(*firstChildItr).m_Description.s_Memory +=
                            (*childItr)->usage();
                        impl(*firstChildItr).m_Description.s_Unused +=
                            (*childItr)->unusage();
                        childItr = m_Children.erase(childItr);
                    }
                    impl(*firstChildItr).m_Description.s_Name +=
                        " [*" + std::to_string(count) + "]";
                }
            }
        }

        for (auto& child : m_Children) {
            child->compress();
        }
    }

private:
    using TMemoryUsagePtrList = std::list<TMemoryUsagePtr>;
    using TMemoryUsageVec = std::vector<SMemoryUsage>;

private:
    //! Give out data to the JSON writer to format, recursively
    void summary(CMemoryUsageJsonWriter& writer) const {
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
                impl(child).summary(writer);
            }
            writer.endArray();
        }

        writer.endObject();
    }

private:
    //! Collection of child items
    TMemoryUsagePtrList m_Children;

    //! Collection of component items within this node
    TMemoryUsageVec m_Items;

    //! Description of this item
    SMemoryUsage m_Description;
};

CMemoryUsage::CMemoryUsage() : m_Impl{std::make_unique<CImpl>()} {
}
CMemoryUsage::~CMemoryUsage() = default;

CMemoryUsage::TMemoryUsagePtr CMemoryUsage::addChild() {
    return m_Impl->addChild();
}

CMemoryUsage::TMemoryUsagePtr CMemoryUsage::addChild(std::size_t initialAmount) {
    return m_Impl->addChild(initialAmount);
}

void CMemoryUsage::addItem(const SMemoryUsage& item) {
    m_Impl->addItem(item);
}

void CMemoryUsage::addItem(const std::string& name, std::size_t memory) {
    m_Impl->addItem(name, memory);
}

void CMemoryUsage::setName(const SMemoryUsage& item) {
    m_Impl->setName(item);
}

void CMemoryUsage::setName(const std::string& name, std::size_t memory) {
    m_Impl->setName(name, memory);
}

void CMemoryUsage::setName(const std::string& name) {
    m_Impl->setName(name);
}

std::size_t CMemoryUsage::usage() const {
    return m_Impl->usage();
}

std::size_t CMemoryUsage::unusage() const {
    return m_Impl->unusage();
}

void CMemoryUsage::compress() {
    m_Impl->compress();
}

void CMemoryUsage::print(std::ostream& outStream) const {
    m_Impl->print(outStream);
}
}
}
