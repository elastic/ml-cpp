/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#include <maths/CClusterer.h>

namespace ml {
namespace maths {
namespace {
const std::string INDEX_TAG("a");
}

CClustererTypes::CIndexGenerator::CIndexGenerator() : m_IndexHeap(new TSizeVec(1u, 0u)) {
}

bool CClustererTypes::CIndexGenerator::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    m_IndexHeap->clear();

    do {
        if (core::CPersistUtils::restore(INDEX_TAG, *m_IndexHeap, traverser) == false) {
            LOG_ERROR(<< "Invalid indices in " << traverser.value());
            return false;
        }
    } while (traverser.next());

    return true;
}

void CClustererTypes::CIndexGenerator::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(INDEX_TAG, *m_IndexHeap, inserter);
}

CClustererTypes::CIndexGenerator CClustererTypes::CIndexGenerator::deepCopy() const {
    CIndexGenerator result;
    result.m_IndexHeap.reset(new TSizeVec(*m_IndexHeap));
    return result;
}

std::size_t CClustererTypes::CIndexGenerator::next() const {
    std::size_t result = m_IndexHeap->front();
    std::pop_heap(m_IndexHeap->begin(), m_IndexHeap->end(), std::greater<std::size_t>());
    m_IndexHeap->pop_back();
    if (m_IndexHeap->empty()) {
        m_IndexHeap->push_back(result + 1u);
    }
    return result;
}

void CClustererTypes::CIndexGenerator::recycle(std::size_t index) {
    m_IndexHeap->push_back(index);
    std::push_heap(m_IndexHeap->begin(), m_IndexHeap->end(), std::greater<std::size_t>());
}

void CClustererTypes::CIndexGenerator::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CClusterer::CIndexGenerator");
    core::CMemoryDebug::dynamicSize("m_IndexHeap", m_IndexHeap, mem);
}

std::size_t CClustererTypes::CIndexGenerator::memoryUsage() const {
    std::size_t mem = core::CMemory::dynamicSize(m_IndexHeap);
    return mem;
}

std::string CClustererTypes::CIndexGenerator::print() const {
    return core::CContainerPrinter::print(*m_IndexHeap);
}

const std::string CClustererTypes::X_MEANS_ONLINE_1D_TAG("a");
const std::string CClustererTypes::K_MEANS_ONLINE_1D_TAG("b");
const std::string CClustererTypes::X_MEANS_ONLINE_TAG("c");
}
}
