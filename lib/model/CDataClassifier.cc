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

#include <model/CDataClassifier.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/CStringUtils.h>
#include <core/RestoreMacros.h>

#include <maths/common/CIntegerTools.h>

#include <numeric>

namespace ml {
namespace model {
namespace {
const double EPS{10.0 * std::numeric_limits<double>::epsilon()};
const std::string IS_INTEGER_TAG{"a"};
const std::string IS_NON_NEGATIVE_TAG{"b"};
std::string EMPTY_STRING;
}

void CDataClassifier::add(model_t::EFeature feature, double value, unsigned int count) {
    m_IsNonNegative = m_IsNonNegative && value >= 0.0;
    if (m_IsInteger) {
        if (model_t::isMeanFeature(feature)) {
            value *= count;
        }
        m_IsInteger = maths::common::CIntegerTools::isInteger(value, EPS * value);
    }
}

void CDataClassifier::add(model_t::EFeature feature, const TDouble1Vec& values, unsigned int count) {
    for (const auto& value : values) {
        this->add(feature, value, count);
    }
}

bool CDataClassifier::isInteger() const {
    return m_IsInteger;
}

bool CDataClassifier::isNonNegative() const {
    return m_IsNonNegative;
}

void CDataClassifier::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(IS_INTEGER_TAG, static_cast<int>(m_IsInteger));
    inserter.insertValue(IS_NON_NEGATIVE_TAG, static_cast<int>(m_IsNonNegative));
}

bool CDataClassifier::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_BOOL(IS_INTEGER_TAG, m_IsInteger)
        RESTORE_BOOL(IS_NON_NEGATIVE_TAG, m_IsNonNegative)
    } while (traverser.next());
    return true;
}
}
}
