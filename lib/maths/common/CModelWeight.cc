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

#include <maths/common/CModelWeight.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/CTools.h>

#include <cmath>

namespace ml {
namespace maths {
namespace common {
namespace {

// We use short field names to reduce the state size
const core::TPersistenceTag LOG_WEIGHT_TAG("a", "log_weight");
const core::TPersistenceTag LONG_TERM_LOG_WEIGHT_TAG("c", "long_term_log_weight");

const double LOG_SMALLEST_WEIGHT = std::log(CTools::smallestProbability());
}

CModelWeight::CModelWeight(double weight)
    : m_LogWeight(std::log(weight)), m_LongTermLogWeight(m_LogWeight) {
}

CModelWeight::operator double() const {
    return m_LogWeight < LOG_SMALLEST_WEIGHT ? 0.0 : std::exp(m_LogWeight);
}

double CModelWeight::logWeight() const {
    return m_LogWeight;
}

void CModelWeight::logWeight(double logWeight) {
    m_LogWeight = logWeight;
}

void CModelWeight::addLogFactor(double logFactor) {
    m_LogWeight += logFactor;
}

void CModelWeight::age(double alpha) {
    m_LogWeight = alpha * m_LogWeight + (1 - alpha) * m_LongTermLogWeight;
}

uint64_t CModelWeight::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_LogWeight);
    return CChecksum::calculate(seed, m_LongTermLogWeight);
}

bool CModelWeight::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(LOG_WEIGHT_TAG, m_LogWeight)
        RESTORE_BUILT_IN(LONG_TERM_LOG_WEIGHT_TAG, m_LongTermLogWeight)
    } while (traverser.next());
    return true;
}

void CModelWeight::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(LOG_WEIGHT_TAG, m_LogWeight, core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(LONG_TERM_LOG_WEIGHT_TAG, m_LongTermLogWeight,
                         core::CIEEE754::E_SinglePrecision);
}
}
}
}
