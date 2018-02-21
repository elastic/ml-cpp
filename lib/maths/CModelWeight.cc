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

#include <maths/CModelWeight.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CTools.h>

#include <math.h>

namespace ml
{
namespace maths
{

namespace
{

// We use short field names to reduce the state size
const std::string LOG_WEIGHT_TAG("a");
const std::string LONG_TERM_LOG_WEIGHT_TAG("c");
const double LOG_SMALLEST_WEIGHT = ::log(CTools::smallestProbability());

}

CModelWeight::CModelWeight(double weight) :
    m_LogWeight(::log(weight)),
    m_LongTermLogWeight(m_LogWeight)
{}

CModelWeight::operator double(void) const
{
    return m_LogWeight < LOG_SMALLEST_WEIGHT ? 0.0 : ::exp(m_LogWeight);
}

double CModelWeight::logWeight(void) const
{
    return m_LogWeight;
}

void CModelWeight::logWeight(double logWeight)
{
    m_LogWeight = logWeight;
}

void CModelWeight::addLogFactor(double logFactor)
{
    m_LogWeight += logFactor;
}

void CModelWeight::age(double alpha)
{
    m_LogWeight = alpha * m_LogWeight + (1 - alpha) * m_LongTermLogWeight;
}

uint64_t CModelWeight::checksum(uint64_t seed) const
{
    seed = CChecksum::calculate(seed, m_LogWeight);
    return CChecksum::calculate(seed, m_LongTermLogWeight);
}

bool CModelWeight::acceptRestoreTraverser(core::CStateRestoreTraverser &traverser)
{
    do
    {
        const std::string &name = traverser.name();
        RESTORE_BUILT_IN(LOG_WEIGHT_TAG, m_LogWeight)
        RESTORE_BUILT_IN(LONG_TERM_LOG_WEIGHT_TAG, m_LongTermLogWeight)
    }
    while (traverser.next());
    return true;
}

void CModelWeight::acceptPersistInserter(core::CStatePersistInserter &inserter) const
{
    inserter.insertValue(LOG_WEIGHT_TAG,
                         m_LogWeight,
                         core::CIEEE754::E_DoublePrecision);
    inserter.insertValue(LONG_TERM_LOG_WEIGHT_TAG,
                         m_LongTermLogWeight,
                         core::CIEEE754::E_SinglePrecision);
}

}
}
