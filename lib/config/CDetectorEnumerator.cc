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

#include <config/CDetectorEnumerator.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <config/CDetectorSpecification.h>
#include <config/Constants.h>

#include <boost/range.hpp>
#include <boost/ref.hpp>

namespace ml {
namespace config {

namespace {

typedef std::vector<std::string> TStrVec;
typedef boost::reference_wrapper<const TStrVec> TStrVecCRef;
typedef std::vector<TStrVecCRef> TStrVecCRefVec;

//! Add detectors for the partitioning fields \p candidates.
void add(std::size_t p,
         const std::size_t* indices,
         const TStrVecCRef* candidates,
         std::size_t a,
         std::size_t b,
         CDetectorEnumerator::TDetectorSpecificationVec& result) {
    LOG_TRACE("a = " << a << " b = " << b);
    if (a == b) {
        return;
    }

    for (std::size_t i = 0u; i < p; ++i) {
        for (std::size_t j = a; j < b; ++j) {
            const TStrVec& ci = candidates[i];
            LOG_TRACE("candidates = " << core::CContainerPrinter::print(ci));

            for (std::size_t k = 0u; k < ci.size(); ++k) {
                if (result[j].canAddPartitioning(indices[i], ci[k])) {
                    std::size_t id = result.size();
                    result.push_back(CDetectorSpecification(result[j]));
                    result.back().id(id);
                    result.back().addPartitioning(indices[i], ci[k]);
                }
            }
        }
    }
}
}

CDetectorEnumerator::CDetectorEnumerator(const CAutoconfigurerParams& params) : m_Params(params) {
}

void CDetectorEnumerator::addFunction(config_t::EFunctionCategory category) {
    m_Functions.push_back(category);
}

void CDetectorEnumerator::addCategoricalFunctionArgument(const std::string& argument) {
    m_CandidateCategoricalFunctionArguments.push_back(argument);
}

void CDetectorEnumerator::addMetricFunctionArgument(const std::string& argument) {
    m_CandidateMetricFunctionArguments.push_back(argument);
}

void CDetectorEnumerator::addByField(const std::string& by) {
    m_CandidateByFields.push_back(by);
}

void CDetectorEnumerator::addRareByField(const std::string& by) {
    m_CandidateRareByFields.push_back(by);
}

void CDetectorEnumerator::addOverField(const std::string& over) {
    m_CandidateOverFields.push_back(over);
}

void CDetectorEnumerator::addPartitionField(const std::string& partition) {
    m_CandidatePartitionFields.push_back(partition);
}

void CDetectorEnumerator::generate(TDetectorSpecificationVec& result) {
    result.clear();

    this->generateNoPartitioning(result);
    std::size_t n0 = result.size();

    this->addOnePartitioning(0, n0, result);
    std::size_t n1 = result.size();

    this->addTwoPartitioning(n0, n1, result);
    std::size_t n2 = result.size();

    this->addThreePartitioning(n1, n2, result);
}

void CDetectorEnumerator::generateNoPartitioning(TDetectorSpecificationVec& result) const {
    for (std::size_t i = 0u; i < m_Functions.size(); ++i) {
        config_t::EFunctionCategory function = m_Functions[i];

        if (config_t::isRare(function)) {
            continue;
        }

        try {
            std::size_t id = result.size();

            if (config_t::hasArgument(function)) {
                const TStrVec& arguments = config_t::isMetric(function)
                                               ? m_CandidateMetricFunctionArguments
                                               : m_CandidateCategoricalFunctionArguments;
                for (std::size_t j = 0u; j < arguments.size(); ++j) {
                    result.push_back(CDetectorSpecification(m_Params, function, arguments[j], id));
                }
            } else {
                result.push_back(CDetectorSpecification(m_Params, function, id));
            }
        } catch (std::exception& e) { LOG_ERROR("Bad detector: " << e.what()); }
    }
}

void CDetectorEnumerator::addOnePartitioning(std::size_t a,
                                             std::size_t b,
                                             TDetectorSpecificationVec& result) const {
    TStrVecCRef candidates[] = {boost::cref(m_CandidateByFields),
                                boost::cref(m_CandidateOverFields),
                                boost::cref(m_CandidatePartitionFields)};
    add(boost::size(constants::CFieldIndices::PARTITIONING),
        constants::CFieldIndices::PARTITIONING,
        candidates,
        a,
        b,
        result);

    for (std::size_t i = 0u; i < m_Functions.size(); ++i) {
        config_t::EFunctionCategory function = m_Functions[i];
        if (config_t::isRare(function)) {
            try {
                for (std::size_t j = 0u; j < m_CandidateRareByFields.size(); ++j) {
                    std::size_t id = result.size();
                    result.push_back(CDetectorSpecification(m_Params, function, id));
                    result.back().addPartitioning(constants::BY_INDEX, m_CandidateRareByFields[j]);
                }
            } catch (std::exception& e) { LOG_ERROR("Bad detector: " << e.what()); }
        }
    }
}

void CDetectorEnumerator::addTwoPartitioning(std::size_t a,
                                             std::size_t b,
                                             TDetectorSpecificationVec& result) const {
    static std::size_t OVER_AND_PARTITION[] = {constants::OVER_INDEX, constants::PARTITION_INDEX};
    TStrVecCRef candidates[] = {boost::cref(m_CandidateOverFields),
                                boost::cref(m_CandidatePartitionFields)};
    add(boost::size(OVER_AND_PARTITION), OVER_AND_PARTITION, candidates, a, b, result);
}

void CDetectorEnumerator::addThreePartitioning(std::size_t a,
                                               std::size_t b,
                                               TDetectorSpecificationVec& result) const {
    static std::size_t PARTITION[] = {constants::PARTITION_INDEX};
    TStrVecCRef candidates[] = {boost::cref(m_CandidatePartitionFields)};
    add(boost::size(PARTITION), PARTITION, candidates, a, b, result);
}
}
}
