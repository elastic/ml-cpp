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

#include <model/CSample.h>

#include <core/CIEEE754.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CStringUtils.h>

#include <maths/CChecksum.h>

#include <model/CFeatureData.h>
#include <model/ModelTypes.h>

namespace ml {
namespace model {

std::string CSample::SToString::operator()(const CSample& sample) const {
    std::string result =
        core::CStringUtils::typeToString(sample.m_Time) + core::CPersistUtils::PAIR_DELIMITER +
        core::CStringUtils::typeToStringPrecise(sample.m_VarianceScale,
                                                core::CIEEE754::E_SinglePrecision) +
        core::CPersistUtils::PAIR_DELIMITER +
        core::CStringUtils::typeToStringPrecise(sample.m_Count, core::CIEEE754::E_SinglePrecision);
    for (std::size_t i = 0u; i < sample.m_Value.size(); ++i) {
        result += core::CPersistUtils::PAIR_DELIMITER +
                  core::CStringUtils::typeToStringPrecise(sample.m_Value[i],
                                                          core::CIEEE754::E_SinglePrecision);
    }
    return result;
}

bool CSample::SFromString::operator()(const std::string& token, CSample& value) const {
    core::CStringUtils::TStrVec tokens;
    std::string remainder;
    core::CStringUtils::tokenise(std::string(1, core::CPersistUtils::PAIR_DELIMITER),
                                 token,
                                 tokens,
                                 remainder);
    if (!remainder.empty()) {
        tokens.push_back(remainder);
    }

    if (!core::CStringUtils::stringToType(tokens[0], value.m_Time) ||
        !core::CStringUtils::stringToType(tokens[1], value.m_VarianceScale) ||
        !core::CStringUtils::stringToType(tokens[2], value.m_Count)) {
        LOG_ERROR("Cannot parse as sample: " << token);
        return false;
    }
    for (std::size_t i = 3u; i < tokens.size(); ++i) {
        double vi;
        if (!core::CStringUtils::stringToType(tokens[i], vi)) {
            LOG_ERROR("Cannot parse as sample: " << token);
            return false;
        }
        value.m_Value.push_back(vi);
    }
    return true;
}

CSample::CSample(void) : m_Time(0), m_Value(), m_VarianceScale(0.0), m_Count(0) {}

CSample::CSample(core_t::TTime time, const TDouble1Vec& value, double varianceScale, double count)
    : m_Time(time), m_Value(value), m_VarianceScale(varianceScale), m_Count(count) {}

CSample::TDouble1Vec CSample::value(std::size_t dimension) const {
    typedef std::vector<std::size_t> TSizeVec;

    TDouble1Vec result;
    const TSizeVec& indices = CFeatureDataIndexing::valueIndices(dimension);
    result.reserve(indices.size());
    for (std::size_t i = 0u; i < indices.size(); ++i) {
        result.push_back(m_Value[indices[i]]);
    }
    return result;
}

uint64_t CSample::checksum(void) const {
    uint64_t seed = static_cast<uint64_t>(m_Time);
    seed = maths::CChecksum::calculate(seed, m_Value);
    seed = maths::CChecksum::calculate(seed, m_VarianceScale);
    return maths::CChecksum::calculate(seed, m_Count);
}

std::string CSample::print(void) const {
    std::ostringstream result;
    result << '(' << m_Time << ' ' << core::CContainerPrinter::print(m_Value) << ' '
           << m_VarianceScale << ' ' << m_Count << ')';
    return result.str();
}

void CSample::debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
    mem->setName("CSample");
    core::CMemoryDebug::dynamicSize("m_Value", m_Value, mem);
}

std::size_t CSample::memoryUsage(void) const {
    return core::CMemory::dynamicSize(m_Value);
}
}
}
