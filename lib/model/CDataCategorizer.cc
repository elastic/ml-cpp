/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CDataCategorizer.h>

#include <core/CMemory.h>
#include <core/CStateRestoreTraverser.h>

#include <model/CLimits.h>

namespace ml {
namespace model {

// Initialise statics
const int CDataCategorizer::SOFT_CATEGORIZATION_FAILURE_ERROR{-1};
const int CDataCategorizer::HARD_CATEGORIZATION_FAILURE_ERROR{-2};
const CDataCategorizer::TStrStrUMap CDataCategorizer::EMPTY_FIELDS;

CDataCategorizer::CDataCategorizer(CLimits& limits, const std::string& fieldName)
    : m_Limits{limits}, m_FieldName{fieldName},
      m_ExamplesCollector{limits.maxExamples()}, m_LastPersistTime{0} {
    m_Limits.resourceMonitor().registerComponent(*this);
}

CDataCategorizer::~CDataCategorizer() {
    m_Limits.resourceMonitor().unRegisterComponent(*this);
}

int CDataCategorizer::computeCategory(bool isDryRun, const std::string& str, std::size_t rawStringLen) {
    return this->computeCategory(isDryRun, EMPTY_FIELDS, str, rawStringLen);
}

const std::string& CDataCategorizer::fieldName() const {
    return m_FieldName;
}

core_t::TTime CDataCategorizer::lastPersistTime() const {
    return m_LastPersistTime;
}

void CDataCategorizer::lastPersistTime(core_t::TTime lastPersistTime) {
    m_LastPersistTime = lastPersistTime;
}

void CDataCategorizer::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CDataCategorizer");
    core::CMemoryDebug::dynamicSize("m_FieldName", m_FieldName, mem);
    core::CMemoryDebug::dynamicSize("m_ExamplesCollector", m_ExamplesCollector, mem);
}

std::size_t CDataCategorizer::memoryUsage() const {
    std::size_t mem = 0;
    mem += core::CMemory::dynamicSize(m_FieldName);
    mem += core::CMemory::dynamicSize(m_ExamplesCollector);
    return mem;
}

bool CDataCategorizer::addExample(int categoryId, const std::string& example) {
    // Don't add examples if we're memory-limited
    if (m_Limits.resourceMonitor().getMemoryStatus() != model_t::E_MemoryStatusOk) {
        LOG_TRACE(<< "Not adding example as memory status is "
                  << m_Limits.resourceMonitor().getMemoryStatus());
        return false;
    }
    return m_ExamplesCollector.add(categoryId, example);
}

const CCategoryExamplesCollector& CDataCategorizer::examplesCollector() const {
    return m_ExamplesCollector;
}

bool CDataCategorizer::areNewCategoriesAllowed() {
    return m_Limits.resourceMonitor().areAllocationsAllowed();
}

bool CDataCategorizer::restoreExamplesCollector(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel(
            std::bind(&model::CCategoryExamplesCollector::acceptRestoreTraverser,
                      std::ref(m_ExamplesCollector), std::placeholders::_1)) == false ||
        traverser.haveBadState()) {
        LOG_ERROR(<< "Cannot restore category examples, unexpected element: "
                  << traverser.value());
        return false;
    }
    return true;
}
}
}
