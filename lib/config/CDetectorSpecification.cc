/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/CDetectorSpecification.h>

#include <core/CLogger.h>

#include <maths/COrderings.h>

#include <config/CAutoconfigurerParams.h>
#include <config/CFieldStatistics.h>
#include <config/CPenalty.h>
#include <config/Constants.h>

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/range.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace ml {
namespace config {
namespace {

using TSizeVec = std::vector<std::size_t>;
using TDoubleVec = std::vector<double>;

//! \brief Checks if the name of some statistics matches a specified value.
class CNameEquals {
public:
    CNameEquals(const std::string& value) : m_Value(&value) {}

    bool operator()(const CFieldStatistics& stats) const {
        return stats.name() == *m_Value;
    }

private:
    const std::string* m_Value;
};

//! Extract the full function name from the configuration.
std::string fullFunctionName(config_t::ESide side,
                             CDetectorSpecification::EFuzzyBool ignoreEmpty,
                             bool isPopulation,
                             config_t::EFunctionCategory function) {
    std::string result;
    switch (side) {
    case config_t::E_LowSide:
        result += "low_";
        break;
    case config_t::E_HighSide:
        result += "high_";
        break;
    case config_t::E_TwoSide:
        break;
    case config_t::E_UndeterminedSide:
        result += "[low_|high_]";
        break;
    }
    if (!isPopulation && config_t::hasDoAndDontIgnoreEmptyVersions(function)) {
        switch (ignoreEmpty) {
        case CDetectorSpecification::E_True:
            result += function == config_t::E_Count ? "non_zero_" : "non_null_";
            break;
        case CDetectorSpecification::E_False:
            break;
        case CDetectorSpecification::E_Maybe:
            result += function == config_t::E_Count ? "[non_zero_]" : "[non_null_]";
            break;
        }
    }
    result += config_t::print(function);
    return result;
}

//! Get the maximum penalty indexed by \p indices.
double maxPenalty(const TSizeVec& indices, const TDoubleVec& penalties) {
    double result = 0.0;
    for (std::size_t i = 0u; i < indices.size(); ++i) {
        result = std::max(result, penalties[indices[i]]);
    }
    return result;
}

//! Set the penalties indexed by \p indices to \p value.
void fill(const TSizeVec& indices, double value, TDoubleVec& penalties) {
    for (std::size_t i = 0u; i < indices.size(); ++i) {
        penalties[indices[i]] = value;
    }
}

//! Mapping from ignore empty unique identifer to its value.
const bool IGNORE_EMPTY[] = {false, true};

//! Get the ignore empty unique identifier.
std::size_t ignoreEmptyId(bool ignoreEmpty) {
    return std::find(std::begin(IGNORE_EMPTY), std::end(IGNORE_EMPTY), ignoreEmpty) -
           std::begin(IGNORE_EMPTY);
}
}

CDetectorSpecification::CDetectorSpecification(const CAutoconfigurerParams& params,
                                               config_t::EFunctionCategory function,
                                               std::size_t id)
    : m_Params(params), m_Function(function),
      m_Side(config_t::hasSidedCalculation(function) ? config_t::E_UndeterminedSide
                                                     : config_t::E_TwoSide),
      m_IgnoreEmpty(config_t::hasDoAndDontIgnoreEmptyVersions(function) ? E_Maybe : E_True),
      m_Penalties(2 * params.candidateBucketLengths().size()),
      m_PenaltyDescriptions(2 * params.candidateBucketLengths().size()),
      m_Id(id), m_CountStatistics(nullptr) {
    this->initializePenalties();
    if (config_t::hasArgument(function)) {
        throw std::logic_error(std::string("No argument supplied for '") +
                               config_t::print(function) + "'");
    }
    std::fill_n(m_FieldStatistics, constants::NUMBER_FIELD_INDICES, nullptr);
}

CDetectorSpecification::CDetectorSpecification(const CAutoconfigurerParams& params,
                                               config_t::EFunctionCategory function,
                                               const std::string& argument,
                                               std::size_t id)
    : m_Params(params), m_Function(function),
      m_Side(config_t::hasSidedCalculation(function) ? config_t::E_UndeterminedSide
                                                     : config_t::E_TwoSide),
      m_IgnoreEmpty(config_t::hasDoAndDontIgnoreEmptyVersions(function) ? E_Maybe : E_True),
      m_Penalties(2 * params.candidateBucketLengths().size()),
      m_PenaltyDescriptions(2 * params.candidateBucketLengths().size()),
      m_Id(id), m_CountStatistics(nullptr) {
    this->initializePenalties();
    if (!config_t::hasArgument(function)) {
        LOG_ERROR(<< "Ignoring argument '" + argument + "' for '" +
                         config_t::print(function) + "'");
    } else {
        m_FunctionFields[constants::ARGUMENT_INDEX] = argument;
    }
    std::fill_n(m_FieldStatistics, constants::NUMBER_FIELD_INDICES, nullptr);
}

void CDetectorSpecification::swap(CDetectorSpecification& other) {
    std::swap(m_Params, other.m_Params);
    std::swap(m_Function, other.m_Function);
    std::swap(m_Side, other.m_Side);
    std::swap(m_IgnoreEmpty, other.m_IgnoreEmpty);
    m_Influencers.swap(other.m_Influencers);
    m_BucketLength.swap(other.m_BucketLength);
    m_Penalties.swap(other.m_Penalties);
    m_PenaltyDescriptions.swap(other.m_PenaltyDescriptions);
    std::swap(m_Penalty, other.m_Penalty);
    std::swap(m_Id, other.m_Id);
    for (std::size_t i = 0u; i < constants::NUMBER_FIELD_INDICES; ++i) {
        m_FunctionFields[i].swap(other.m_FunctionFields[i]);
        std::swap(m_FieldStatistics[i], other.m_FieldStatistics[i]);
    }
    std::swap(m_CountStatistics, other.m_CountStatistics);
}

void CDetectorSpecification::side(config_t::ESide side) {
    m_Side = side;
}

void CDetectorSpecification::ignoreEmpty(bool ignoreEmpty) {
    m_IgnoreEmpty = ignoreEmpty ? E_True : E_False;
}

bool CDetectorSpecification::canAddPartitioning(std::size_t index,
                                                const std::string& value) const {
    // Rules:
    //   1) We can only add a field to a detector whose index is greater
    //      than any field currently set.
    //   2) We can't have duplicate fields.

    return static_cast<int>(index) > this->highestFieldIndex() &&
           std::find(std::begin(m_FunctionFields), std::end(m_FunctionFields),
                     value) == std::end(m_FunctionFields);
}

void CDetectorSpecification::addPartitioning(std::size_t index, const std::string& value) {
    m_FunctionFields[index] = value;
    if (index == constants::OVER_INDEX) {
        m_IgnoreEmpty = E_True;
    }
}

void CDetectorSpecification::addInfluencer(const std::string& influencer) {
    std::size_t n = m_Influencers.size();
    m_Influencers.push_back(influencer);
    if (n > 0) {
        std::inplace_merge(m_Influencers.begin(), m_Influencers.begin() + n,
                           m_Influencers.end());
    }
}

void CDetectorSpecification::bucketLength(core_t::TTime bucketLength) {
    m_BucketLength = bucketLength;
}

void CDetectorSpecification::addFieldStatistics(const TFieldStatisticsVec& stats) {
    for (std::size_t i = 0u; i < boost::size(constants::CFieldIndices::ALL); ++i) {
        if (const TOptionalStr& field = m_FunctionFields[constants::CFieldIndices::ALL[i]]) {
            m_FieldStatistics[constants::CFieldIndices::ALL[i]] =
                &(*std::find_if(stats.begin(), stats.end(), CNameEquals(*field)));
        }
    }
}

void CDetectorSpecification::setCountStatistics(const CDataCountStatistics& stats) {
    m_CountStatistics = &stats;
}

void CDetectorSpecification::setPenalty(const TPenaltyPtr& penalty) {
    m_Penalty = penalty;
}

double CDetectorSpecification::score() const {
    TSizeVecCPtrAry indicesInUse = this->penaltyIndicesInUse();
    double penalty = 0.0;
    for (auto indexInUse : indicesInUse) {
        penalty = std::max(penalty, maxPenalty(*indexInUse, m_Penalties));
    }
    return CPenalty::score(penalty);
}

void CDetectorSpecification::scores(TParamScoresVec& result) const {
    result.reserve(m_Penalties.size());
    const TTimeVec& candidates = this->params().candidateBucketLengths();
    for (std::size_t iid = 0u; iid < boost::size(IGNORE_EMPTY); ++iid) {
        for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
            std::size_t pid = this->params().penaltyIndexFor(bid, IGNORE_EMPTY[iid]);
            double score = CPenalty::score(m_Penalties[pid]);
            const TStrVec& descriptions = m_PenaltyDescriptions[pid];
            if (score > this->params().minimumDetectorScore()) {
                const std::string& name = config_t::ignoreEmptyVersionName(
                    m_Function, IGNORE_EMPTY[iid], this->isPopulation());
                result.push_back(SParamScores(candidates[bid], name, score, descriptions));
            }
        }
    }
}

void CDetectorSpecification::applyPenalty(double penalty, const std::string& description) {
    LOG_TRACE(<< "penalty = " << penalty);
    if (penalty == 1.0) {
        return;
    }
    for (std::size_t i = 0u; i < m_Penalties.size(); ++i) {
        m_Penalties[i] *= penalty;
        if (!description.empty()) {
            m_PenaltyDescriptions[i].push_back(description);
        }
    }
}

void CDetectorSpecification::applyPenalties(const TSizeVec& indices,
                                            const TDoubleVec& penalties,
                                            const TStrVec& descriptions) {
    LOG_TRACE(<< "penalties = " << core::CContainerPrinter::print(penalties));
    for (std::size_t i = 0u; i < indices.size(); ++i) {
        if (penalties[i] == 1.0) {
            continue;
        }
        m_Penalties[indices[i]] *= penalties[i];
        if (!descriptions[i].empty()) {
            m_PenaltyDescriptions[indices[i]].push_back(descriptions[i]);
        }
    }
    LOG_TRACE(<< "cumulative = " << core::CContainerPrinter::print(m_Penalties));
}

void CDetectorSpecification::refreshScores() {
    LOG_TRACE(<< "*** Refreshing scores ***");
    this->initializePenalties();
    m_Penalty->penalize(*this);
    this->refreshIgnoreEmpty();
}

config_t::EFunctionCategory CDetectorSpecification::function() const {
    return m_Function;
}

const CDetectorSpecification::TOptionalStr& CDetectorSpecification::argumentField() const {
    return m_FunctionFields[constants::ARGUMENT_INDEX];
}

const CDetectorSpecification::TOptionalStr& CDetectorSpecification::byField() const {
    return m_FunctionFields[constants::BY_INDEX];
}

const CDetectorSpecification::TOptionalStr& CDetectorSpecification::overField() const {
    return m_FunctionFields[constants::OVER_INDEX];
}

const CDetectorSpecification::TOptionalStr& CDetectorSpecification::partitionField() const {
    return m_FunctionFields[constants::PARTITION_INDEX];
}

const CDetectorSpecification::TStrVec& CDetectorSpecification::influences() const {
    return m_Influencers;
}

void CDetectorSpecification::candidateBucketLengths(TTimeVec& result) const {
    const TTimeVec& candidates = this->params().candidateBucketLengths();
    result.reserve(candidates.size());
    for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
        if (CPenalty::score(maxPenalty(this->params().penaltyIndicesFor(bid),
                                       m_Penalties)) > 0.0) {
            result.push_back(candidates[bid]);
        }
    }
}

bool CDetectorSpecification::isPopulation() const {
    return static_cast<bool>(this->overField());
}

bool CDetectorSpecification::operator<(const CDetectorSpecification& rhs) const {
#define LESS(lhs, rhs)                                                         \
    if (lhs < rhs)                                                             \
        return true;                                                           \
    if (rhs < lhs)                                                             \
    return false

    LESS(m_Function, rhs.m_Function);
    LESS(m_Side, rhs.m_Side);
    LESS(m_IgnoreEmpty, rhs.m_IgnoreEmpty);

    maths::COrderings::SOptionalLess less;
    if (less(m_BucketLength, rhs.m_BucketLength)) {
        return true;
    }
    if (less(rhs.m_BucketLength, m_BucketLength)) {
        return false;
    }

    if (std::lexicographical_compare(std::begin(m_FunctionFields), std::end(m_FunctionFields),
                                     std::begin(rhs.m_FunctionFields),
                                     std::end(rhs.m_FunctionFields), less)) {
        return true;
    }
    if (std::lexicographical_compare(
            std::begin(rhs.m_FunctionFields), std::end(rhs.m_FunctionFields),
            std::begin(m_FunctionFields), std::end(m_FunctionFields), less)) {
        return false;
    }

    return m_Influencers < rhs.m_Influencers;
}

bool CDetectorSpecification::operator==(const CDetectorSpecification& rhs) const {
    return m_Function == rhs.m_Function && m_Side == rhs.m_Side &&
           m_IgnoreEmpty == rhs.m_IgnoreEmpty && m_BucketLength == rhs.m_BucketLength &&
           std::equal(std::begin(m_FunctionFields), std::end(m_FunctionFields),
                      std::begin(rhs.m_FunctionFields)) &&
           m_Influencers == rhs.m_Influencers;
}

std::size_t CDetectorSpecification::id() const {
    return m_Id;
}

void CDetectorSpecification::id(std::size_t id) {
    m_Id = id;
}

const CFieldStatistics* CDetectorSpecification::argumentFieldStatistics() const {
    return m_FieldStatistics[constants::ARGUMENT_INDEX];
}

const CFieldStatistics* CDetectorSpecification::byFieldStatistics() const {
    return m_FieldStatistics[constants::BY_INDEX];
}

const CFieldStatistics* CDetectorSpecification::overFieldStatistics() const {
    return m_FieldStatistics[constants::OVER_INDEX];
}

const CFieldStatistics* CDetectorSpecification::partitionFieldStatistics() const {
    return m_FieldStatistics[constants::PARTITION_INDEX];
}

const CDataCountStatistics* CDetectorSpecification::countStatistics() const {
    return m_CountStatistics;
}

std::string CDetectorSpecification::detectorConfig() const {
    if (!this->params().writeDetectorConfigs()) {
        return "";
    }

    using TDoubleTimePr = std::pair<double, core_t::TTime>;
    using TMaxAccumulator =
        maths::CBasicStatistics::COrderStatisticsStack<TDoubleTimePr, 1, maths::COrderings::SFirstGreater>;

    const TTimeVec& candidates = this->params().candidateBucketLengths();

    TMaxAccumulator best;
    for (std::size_t bid = 0u; bid < candidates.size(); ++bid) {
        const TSizeVec& indices = this->params().penaltyIndicesFor(bid);
        for (std::size_t i = 0u; i < indices.size(); ++i) {
            best.add(std::make_pair(m_Penalties[indices[i]], candidates[bid]));
        }
    }

    std::ostringstream result;
    if (CPenalty::score(best[0].first) > this->params().minimumDetectorScore()) {
        const std::string& newLine = this->params().detectorConfigLineEnding();
        result << "{" << newLine << "  \"analysisConfig\": {" << newLine
               << "    \"bucketSpan\": " << best[0].second << newLine << "  },"
               << newLine << "  \"detectors\": [" << newLine << "    {" << newLine
               << "      \"function\":\"" << config_t::print(m_Function) << "\"";
        if (const CDetectorSpecification::TOptionalStr& argument = this->argumentField()) {
            result << "," << newLine << "      \"fieldName\": \"" << *argument << "\"";
        }
        if (const CDetectorSpecification::TOptionalStr& by = this->byField()) {
            result << "," << newLine << "      \"byFieldName\": \"" << *by << "\"";
        }
        if (const CDetectorSpecification::TOptionalStr& over = this->overField()) {
            result << "," << newLine << "      \"overFieldName\": \"" << *over << "\"";
        }
        if (const CDetectorSpecification::TOptionalStr& partition = this->partitionField()) {
            result << "," << newLine << "      \"partitionFieldName\": \""
                   << *partition << "\"";
        }
        result << newLine << "    }" << newLine << "  ]" << newLine << "}";
    }
    return result.str();
}

std::string CDetectorSpecification::description() const {
    std::ostringstream result;
    result
        << fullFunctionName(m_Side, m_IgnoreEmpty, this->isPopulation(), m_Function)
        << (this->argumentField() ? std::string("(") + *this->argumentField() + ")"
                                  : std::string())
        << (this->byField() ? std::string(" by '") + *this->byField() + "'" : std::string())
        << (this->overField() ? std::string(" over '") + *this->overField() + "'"
                              : std::string())
        << (this->partitionField()
                ? std::string(" partition '") + *this->partitionField() + "'"
                : std::string());
    return result.str();
}

const CAutoconfigurerParams& CDetectorSpecification::params() const {
    return m_Params;
}

int CDetectorSpecification::highestFieldIndex() const {
    int result = -1;
    for (std::size_t i = 0u; i < boost::size(m_FunctionFields); ++i) {
        if (m_FunctionFields[i]) {
            result = static_cast<int>(i);
        }
    }
    return result;
}

CDetectorSpecification::TSizeVecCPtrAry CDetectorSpecification::penaltyIndicesInUse() const {
    static const TSizeVec EMPTY;
    TSizeVecCPtrAry result;
    switch (m_IgnoreEmpty) {
    case E_True:
        result[ignoreEmptyId(true)] = &this->params().penaltyIndicesFor(true);
        result[ignoreEmptyId(false)] = &EMPTY;
        break;
    case E_False:
        result[ignoreEmptyId(true)] = &EMPTY;
        result[ignoreEmptyId(false)] = &this->params().penaltyIndicesFor(false);
        break;
    case E_Maybe:
        result[ignoreEmptyId(true)] = &this->params().penaltyIndicesFor(true);
        result[ignoreEmptyId(false)] = &this->params().penaltyIndicesFor(false);
        break;
    }
    return result;
}

void CDetectorSpecification::initializePenalties() {
    std::fill_n(m_Penalties.begin(), m_Penalties.size(), 0.0);
    TSizeVecCPtrAry indicesInUse = this->penaltyIndicesInUse();
    for (auto indexInUse : indicesInUse) {
        fill(*indexInUse, 1.0, m_Penalties);
    }
    std::fill_n(m_PenaltyDescriptions.begin(), m_PenaltyDescriptions.size(), TStrVec());
}

void CDetectorSpecification::refreshIgnoreEmpty() {
    if (!config_t::hasDoAndDontIgnoreEmptyVersions(m_Function) || this->isPopulation()) {
        return;
    }

    static const EFuzzyBool STATUS[] = {E_Maybe, E_False, E_True, E_Maybe};

    double ptrue = maxPenalty(this->params().penaltyIndicesFor(true), m_Penalties);
    double pfalse = maxPenalty(this->params().penaltyIndicesFor(false), m_Penalties);
    m_IgnoreEmpty =
        STATUS[(CPenalty::score(ptrue) > 0.0 ? 2 : 0) + (CPenalty::score(pfalse) > 0.0 ? 1 : 0)];
}

CDetectorSpecification::SParamScores::SParamScores(core_t::TTime bucketLength,
                                                   const std::string& ignoreEmpty,
                                                   double score,
                                                   const TStrVec& descriptions)
    : s_BucketLength(bucketLength), s_IgnoreEmpty(ignoreEmpty), s_Score(score),
      s_Descriptions(descriptions) {
}
}
}
