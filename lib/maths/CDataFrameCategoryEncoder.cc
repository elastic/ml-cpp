/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CDataFrameCategoryEncoder.h>

#include <core/CContainerPrinter.h>
#include <core/CDataFrame.h>
#include <core/CLogger.h>
#include <core/CPackedBitVector.h>
#include <core/CPersistUtils.h>
#include <core/CTriple.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CDataFrameUtils.h>
#include <maths/COrderings.h>

#include <algorithm>
#include <numeric>

namespace ml {
namespace maths {
namespace {
using TDoubleVec = std::vector<double>;
using TSizeDoublePr = std::pair<std::size_t, double>;
using TSizeDoublePrVec = std::vector<TSizeDoublePr>;
using TSizeDoublePrVecVec = std::vector<TSizeDoublePrVec>;
using TSizeUSet = boost::unordered_set<std::size_t>;
using TDoubleUSet = boost::unordered_set<double>;
using CIdentityEncoding = CDataFrameCategoryEncoder::CIdentityEncoding;
using COneHotEncoding = CDataFrameCategoryEncoder::COneHotEncoding;
using CMappedEncoding = CDataFrameCategoryEncoder::CMappedEncoding;

const std::size_t CATEGORY_FOR_METRICS{std::numeric_limits<std::size_t>::max()};
const std::size_t CATEGORY_FOR_FREQUENCY_ENCODING{CATEGORY_FOR_METRICS - 1};
const std::size_t CATEGORY_FOR_TARGET_MEAN_ENCODING{CATEGORY_FOR_FREQUENCY_ENCODING - 1};
const std::size_t CATEGORY_FOR_DEPENDENT_VARIABLE{CATEGORY_FOR_TARGET_MEAN_ENCODING - 1};

const std::string VERSION_7_5_TAG{"7.5"};

const std::string ENCODING_VECTOR_TAG{"encoding_vector"};
const std::string ENCODING_INPUT_COLUMN_INDEX_TAG{"encoding_input_column_index"};
const std::string ENCODING_MIC_TAG{"encoding_mic"};
const std::string ONE_HOT_ENCODING_CATEGORY_TAG{"one_hot_encoding_category"};
const std::string MAPPED_ENCODING_TYPE_TAG{"mapped_encoding_type"};
const std::string MAPPED_ENCODING_MAP_TAG{"mapped_encoding_map"};
const std::string MAPPED_ENCODING_FALLBACK_TAG{"mapped_encoding_fallback"};
const std::string MAPPED_ENCODING_BINARY_TAG{"mapped_encoding_binary"};
const std::string IDENTITY_ENCODING_TAG{"identity_encoding"};
const std::string ONE_HOT_ENCODING_TAG{"one_hot_encoding"};
const std::string FREQUENCY_ENCODING_TAG{"frequency_encoding"};
const std::string TARGET_MEAN_ENCODING_TAG{"target_mean_encoding"};

// TODO can these constants be replaced with EEncodings?
bool isMetric(std::size_t category) {
    return category == CATEGORY_FOR_METRICS;
}

bool isFrequency(std::size_t category) {
    return category == CATEGORY_FOR_FREQUENCY_ENCODING;
}

bool isTargetMean(std::size_t category) {
    return category == CATEGORY_FOR_TARGET_MEAN_ENCODING;
}

bool isCategory(std::size_t category) {
    return (isMetric(category) || isFrequency(category) || isTargetMean(category)) == false;
}

std::string print(std::size_t category) {
    if (isMetric(category)) {
        return "metric";
    }
    if (isFrequency(category)) {
        return "frequency";
    }
    if (isTargetMean(category)) {
        return "target mean";
    }
    return std::to_string(category);
}

//! \brief Maintains the state for a single feature in a greedy search for the
//! minimum redundancy maximum relevance feature selection.
class CFeatureRelevanceMinusRedundancy {
public:
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

public:
    CFeatureRelevanceMinusRedundancy(std::size_t feature, std::size_t category, double micWithDependentVariable)
        : m_Feature{feature}, m_Category{category}, m_MicWithDependentVariable{
                                                        micWithDependentVariable} {}

    bool isMetric() const { return maths::isMetric(m_Category); }

    bool isFrequency() const { return maths::isFrequency(m_Category); }

    bool isTargetMean() const { return maths::isTargetMean(m_Category); }

    bool isCategory() const { return maths::isCategory(m_Category); }

    std::size_t feature() const { return m_Feature; }

    std::size_t category() const { return m_Category; }

    double micWithDependentVariable() const {
        return m_MicWithDependentVariable;
    }

    double relevanceMinusRedundancy(double redundancyWeight) const {
        return m_MicWithDependentVariable -
               redundancyWeight * this->micWithSelectedVariables();
    }

    std::unique_ptr<CDataFrameUtils::CColumnValue>
    columnValue(const TSizeUSet& rareCategories,
                const TDoubleVec& frequencies,
                const TDoubleVec& targetMeanValues) const {
        if (this->isMetric()) {
            return std::make_unique<CDataFrameUtils::CMetricColumnValue>(m_Feature);
        }
        if (this->isFrequency()) {
            return std::make_unique<CDataFrameUtils::CFrequencyCategoricalColumnValue>(
                m_Feature, frequencies);
        }
        if (this->isTargetMean()) {
            return std::make_unique<CDataFrameUtils::CTargetMeanCategoricalColumnValue>(
                m_Feature, rareCategories, targetMeanValues);
        }
        return std::make_unique<CDataFrameUtils::COneHotCategoricalColumnValue>(
            m_Feature, m_Category);
    }

    void update(const TSizeDoublePrVecVec& mics) {
        auto i = std::find_if(
            mics[m_Feature].begin(), mics[m_Feature].end(),
            [this](const TSizeDoublePr& mic) { return mic.first == m_Category; });
        if (i != mics[m_Feature].end()) {
            m_MicWithSelectedVariables.add(i->second);
        }
    }

private:
    double micWithSelectedVariables() const {
        return CBasicStatistics::mean(m_MicWithSelectedVariables);
    }

private:
    std::size_t m_Feature = 0;
    std::size_t m_Category = 0;
    double m_MicWithDependentVariable = 0.0;
    TMeanAccumulator m_MicWithSelectedVariables;
};

//! \brief Manages a greedy search for the minimum redundancy maximum relevancy
//! feature set.
//!
//! DESCRIPTION:\n
//! Implements a greedy search to approximately solve the optimization problem
//!
//! \f$arg\max_S \frac{1}{|S|} \sum_{f\in S}{MIC(f,t)} - \frac{1}{|S|^2} \sum_{f\in S, g\in S} {MIC(f,g)}\f$
//!
//! This trades redundancy of information in the feature set as a whole and
//! relevance of each individual feature when deciding which to include. We
//! extend the basic measure by including a non-negative redundancy weight
//! which controls the priority of minimizing redundancy vs maximizing relevancy.
//! For more information see
//! https://en.wikipedia.org/wiki/Feature_selection#Minimum-redundancy-maximum-relevance_(mRMR)_feature_selection
//! and references therein.
class CMinRedundancyMaxRelevancyGreedySearch {
public:
    CMinRedundancyMaxRelevancyGreedySearch(double redundancyWeight,
                                           const TSizeDoublePrVecVec& mics)
        : m_RedundancyWeight{redundancyWeight} {
        for (std::size_t i = 0; i < mics.size(); ++i) {
            for (std::size_t j = 0; j < mics[i].size(); ++j) {
                std::size_t category;
                double mic;
                std::tie(category, mic) = mics[i][j];
                if (mic > 0.0) {
                    m_Features.emplace_back(i, category, mic);
                }
            }
        }
    }

    CFeatureRelevanceMinusRedundancy selectNext() {
        auto selected = std::max_element(
            m_Features.begin(), m_Features.end(),
            [this](const CFeatureRelevanceMinusRedundancy& lhs,
                   const CFeatureRelevanceMinusRedundancy& rhs) {
                return lhs.relevanceMinusRedundancy(m_RedundancyWeight) <
                       rhs.relevanceMinusRedundancy(m_RedundancyWeight);
            });
        CFeatureRelevanceMinusRedundancy result{*selected};
        m_Features.erase(selected);
        return result;
    }

    void update(const TSizeDoublePrVecVec& mics) {
        for (auto& feature : m_Features) {
            feature.update(mics);
        }
    }

private:
    using TFeatureRelevanceMinusRedundancyList = std::list<CFeatureRelevanceMinusRedundancy>;

private:
    double m_RedundancyWeight;
    TFeatureRelevanceMinusRedundancyList m_Features;
};
}

CEncodedDataFrameRowRef::CEncodedDataFrameRowRef(const TRowRef& row,
                                                 const CDataFrameCategoryEncoder& encoder)
    : m_Row{row}, m_Encoder{&encoder} {
}

CFloatStorage CEncodedDataFrameRowRef::operator[](std::size_t encodedColumnIndex) const {
    return m_Encoder->encoding(encodedColumnIndex).encode(m_Row);
}

std::size_t CEncodedDataFrameRowRef::index() const {
    return m_Row.index();
}

std::size_t CEncodedDataFrameRowRef::numberColumns() const {
    return m_Encoder->numberEncodedColumns();
}

CDataFrameCategoryEncoder::CDataFrameCategoryEncoder(CMakeDataFrameCategoryEncoder& builder) {
    m_Encodings = builder.makeEncodings();
}

CDataFrameCategoryEncoder::CDataFrameCategoryEncoder(CMakeDataFrameCategoryEncoder&& builder)
    : CDataFrameCategoryEncoder(builder) {
}

CDataFrameCategoryEncoder::CDataFrameCategoryEncoder(core::CStateRestoreTraverser& traverser) {
    if (traverser.traverseSubLevel(std::bind(&CDataFrameCategoryEncoder::acceptRestoreTraverser,
                                             this, std::placeholders::_1)) == false) {
        throw std::runtime_error{"failed to restore category encoder"};
    }
}

CEncodedDataFrameRowRef CDataFrameCategoryEncoder::encode(const TRowRef& row) const {
    return {row, *this};
}

CDataFrameCategoryEncoder::TDoubleVec CDataFrameCategoryEncoder::encodedColumnMics() const {
    TDoubleVec mics;
    mics.reserve(m_Encodings.size());
    for (const auto& encoding : m_Encodings) {
        mics.push_back(encoding->mic());
    }
    return mics;
}

std::size_t CDataFrameCategoryEncoder::numberInputColumns() const {
    // This returns the highest "column index" + 1 of any feature selected as part
    // of encoding and feature selection. For example, this is used to presize arrays
    // containing values associated the features (such as feature importance) which
    // allows direct addressing by the feature column index.
    std::size_t result{0};
    for (const auto& encoding : m_Encodings) {
        result = std::max(result, encoding->inputColumnIndex());
    }
    return result + 1;
}

std::size_t CDataFrameCategoryEncoder::numberEncodedColumns() const {
    return m_Encodings.size();
}

const CDataFrameCategoryEncoder::CEncoding&
CDataFrameCategoryEncoder::encoding(std::size_t encodedColumnIndex) const {
    return *m_Encodings[encodedColumnIndex];
}

bool CDataFrameCategoryEncoder::isBinary(std::size_t encodedColumnIndex) const {
    return m_Encodings[encodedColumnIndex]->isBinary();
}

std::uint64_t CDataFrameCategoryEncoder::checksum(std::uint64_t seed) const {
    return CChecksum::calculate(seed, m_Encodings);
}

void CDataFrameCategoryEncoder::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(VERSION_7_5_TAG, "", inserter);
    inserter.insertLevel(ENCODING_VECTOR_TAG,
                         std::bind(&CDataFrameCategoryEncoder::persistEncodings,
                                   this, std::placeholders::_1));
}

bool CDataFrameCategoryEncoder::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    if (traverser.name() == VERSION_7_5_TAG) {
        do {
            const std::string& name{traverser.name()};
            RESTORE(ENCODING_VECTOR_TAG, traverser.traverseSubLevel(std::bind(
                                             &CDataFrameCategoryEncoder::restoreEncodings,
                                             this, std::placeholders::_1)))
        } while (traverser.next());
        return true;
    }
    LOG_ERROR(<< "Input error: unsupported state serialization version. Currently supported version: "
              << VERSION_7_5_TAG);
    return false;
}

void CDataFrameCategoryEncoder::persistEncodings(core::CStatePersistInserter& inserter) const {
    for (const auto& encoding : m_Encodings) {
        auto persistEncoding = [&encoding](core::CStatePersistInserter& inserter_) {
            encoding->acceptPersistInserter(inserter_);
        };
        inserter.insertLevel(encoding->typeString(), persistEncoding);
    }
}

bool CDataFrameCategoryEncoder::restoreEncodings(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(IDENTITY_ENCODING_TAG, this->restore<CIdentityEncoding>(traverser, 0, 0.0))
        RESTORE(ONE_HOT_ENCODING_TAG, this->restore<COneHotEncoding>(traverser, 0, 0.0, 0))
        RESTORE(FREQUENCY_ENCODING_TAG,
                this->restore<CMappedEncoding>(traverser, 0, 0.0, E_Frequency,
                                               TDoubleVec{}, 0.0))
        RESTORE(TARGET_MEAN_ENCODING_TAG,
                this->restore<CMappedEncoding>(traverser, 0, 0.0, E_TargetMean,
                                               TDoubleVec{}, 0.0))
        LOG_ERROR(<< "Unknown encoding type " << name);
        return false;
    } while (traverser.next());
    return true;
}

template<typename T, typename... Args>
bool CDataFrameCategoryEncoder::restore(core::CStateRestoreTraverser& traverser,
                                        Args&&... args) {
    m_Encodings.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
    if (traverser.traverseSubLevel(std::bind(&T::acceptRestoreTraverser,
                                             static_cast<T*>(m_Encodings.back().get()),
                                             std::placeholders::_1)) == false) {
        LOG_ERROR(<< "Error restoring encoding " << traverser.name());
        return false;
    }
    return true;
}

void CDataFrameCategoryEncoder::accept(CDataFrameCategoryEncoder::CVisitor& visitor) const {
    for (const auto& encoding : m_Encodings) {
        if (encoding->type() == E_IdentityEncoding) {
            visitor.addIdentityEncoding(encoding->inputColumnIndex());
        }
        if (encoding->type() == E_OneHot) {
            auto enc = static_cast<const COneHotEncoding*>(encoding.get());
            visitor.addOneHotEncoding(enc->inputColumnIndex(), enc->hotCategory());
        } else if (encoding->type() == E_Frequency) {
            auto enc = static_cast<const CMappedEncoding*>(encoding.get());
            visitor.addFrequencyEncoding(enc->inputColumnIndex(), enc->map());
        } else if (encoding->type() == E_TargetMean) {
            auto enc = static_cast<const CMappedEncoding*>(encoding.get());
            visitor.addTargetMeanEncoding(enc->inputColumnIndex(), enc->map(),
                                          enc->fallback());
        }
    }
}

CDataFrameCategoryEncoder::CEncoding::CEncoding(std::size_t inputColumnIndex, double mic)
    : m_InputColumnIndex{inputColumnIndex}, m_Mic{mic} {
}

std::size_t CDataFrameCategoryEncoder::CEncoding::inputColumnIndex() const {
    return m_InputColumnIndex;
}

double CDataFrameCategoryEncoder::CEncoding::encode(const TRowRef& row) const {
    return this->encode(row[m_InputColumnIndex]);
}

double CDataFrameCategoryEncoder::CEncoding::mic() const {
    return m_Mic;
}

void CDataFrameCategoryEncoder::CEncoding::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(ENCODING_INPUT_COLUMN_INDEX_TAG, m_InputColumnIndex, inserter);
    core::CPersistUtils::persist(ENCODING_MIC_TAG, m_Mic, inserter);
    this->acceptPersistInserterForDerivedTypeState(inserter);
}

bool CDataFrameCategoryEncoder::CEncoding::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE(ENCODING_INPUT_COLUMN_INDEX_TAG,
                core::CPersistUtils::restore(ENCODING_INPUT_COLUMN_INDEX_TAG,
                                             m_InputColumnIndex, traverser))
        RESTORE(ENCODING_MIC_TAG,
                core::CPersistUtils::restore(ENCODING_MIC_TAG, m_Mic, traverser))
        if (this->acceptRestoreTraverserForDerivedTypeState(traverser) == false) {
            return false;
        }

    } while (traverser.next());
    return true;
}

CDataFrameCategoryEncoder::CIdentityEncoding::CIdentityEncoding(std::size_t inputColumnIndex,
                                                                double mic)
    : CEncoding{inputColumnIndex, mic} {
}

EEncoding CDataFrameCategoryEncoder::CIdentityEncoding::type() const {
    return E_IdentityEncoding;
}

double CDataFrameCategoryEncoder::CIdentityEncoding::encode(double value) const {
    return value;
}

bool CDataFrameCategoryEncoder::CIdentityEncoding::isBinary() const {
    return false;
}

std::uint64_t CDataFrameCategoryEncoder::CIdentityEncoding::checksum() const {
    return CChecksum::calculate(this->inputColumnIndex(), this->mic());
}

const std::string& CDataFrameCategoryEncoder::CIdentityEncoding::typeString() const {
    return IDENTITY_ENCODING_TAG;
}

void CDataFrameCategoryEncoder::CIdentityEncoding::acceptPersistInserterForDerivedTypeState(
    core::CStatePersistInserter& /*inserter*/) const {
    // do nothing
}

bool CDataFrameCategoryEncoder::CIdentityEncoding::acceptRestoreTraverserForDerivedTypeState(
    core::CStateRestoreTraverser& /*traverser*/) {
    return true;
}

CDataFrameCategoryEncoder::COneHotEncoding::COneHotEncoding(std::size_t inputColumnIndex,
                                                            double mic,
                                                            std::size_t hotCategory)
    : CEncoding{inputColumnIndex, mic}, m_HotCategory{hotCategory} {
}

EEncoding CDataFrameCategoryEncoder::COneHotEncoding::type() const {
    return E_OneHot;
}

double CDataFrameCategoryEncoder::COneHotEncoding::encode(double value) const {
    return CDataFrameUtils::isMissing(value)
               ? core::CDataFrame::valueOfMissing()
               : static_cast<std::size_t>(value) == m_HotCategory;
}

bool CDataFrameCategoryEncoder::COneHotEncoding::isBinary() const {
    return true;
}

std::uint64_t CDataFrameCategoryEncoder::COneHotEncoding::checksum() const {
    std::size_t seed{CChecksum::calculate(this->inputColumnIndex(), this->mic())};
    return CChecksum::calculate(seed, m_HotCategory);
}

void CDataFrameCategoryEncoder::COneHotEncoding::acceptPersistInserterForDerivedTypeState(
    core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(ONE_HOT_ENCODING_CATEGORY_TAG, m_HotCategory, inserter);
}

bool CDataFrameCategoryEncoder::COneHotEncoding::acceptRestoreTraverserForDerivedTypeState(
    core::CStateRestoreTraverser& traverser) {

    const std::string& name = traverser.name();
    RESTORE_NO_LOOP(ONE_HOT_ENCODING_CATEGORY_TAG,
                    core::CPersistUtils::restore(ONE_HOT_ENCODING_CATEGORY_TAG,
                                                 m_HotCategory, traverser))
    return true;
}

const std::string& CDataFrameCategoryEncoder::COneHotEncoding::typeString() const {
    return ONE_HOT_ENCODING_TAG;
}

size_t CDataFrameCategoryEncoder::COneHotEncoding::hotCategory() const {
    return m_HotCategory;
}

CDataFrameCategoryEncoder::CMappedEncoding::CMappedEncoding(std::size_t inputColumnIndex,
                                                            double mic,
                                                            EEncoding encoding,
                                                            const TDoubleVec& map,
                                                            double fallback)
    : CEncoding{inputColumnIndex, mic}, m_Encoding{encoding}, m_Map{map}, m_Fallback{fallback} {
    TDoubleUSet uniques{map.begin(), map.end()};
    uniques.insert(m_Fallback);
    m_Binary = uniques.size() == 2;
}

EEncoding CDataFrameCategoryEncoder::CMappedEncoding::type() const {
    return m_Encoding;
}

double CDataFrameCategoryEncoder::CMappedEncoding::encode(double value) const {
    if (CDataFrameUtils::isMissing(value)) {
        return core::CDataFrame::valueOfMissing();
    }
    std::size_t category{static_cast<std::size_t>(value)};
    return category < m_Map.size() ? m_Map[category] : m_Fallback;
}

bool CDataFrameCategoryEncoder::CMappedEncoding::isBinary() const {
    return m_Binary;
}

std::uint64_t CDataFrameCategoryEncoder::CMappedEncoding::checksum() const {
    std::size_t seed{CChecksum::calculate(this->inputColumnIndex(), this->mic())};
    seed = CChecksum::calculate(seed, m_Map);
    seed = CChecksum::calculate(seed, m_Fallback);
    return CChecksum::calculate(seed, m_Binary);
}

void CDataFrameCategoryEncoder::CMappedEncoding::acceptPersistInserterForDerivedTypeState(
    core::CStatePersistInserter& inserter) const {
    core::CPersistUtils::persist(MAPPED_ENCODING_TYPE_TAG, m_Encoding, inserter);
    core::CPersistUtils::persist(MAPPED_ENCODING_MAP_TAG, m_Map, inserter);
    core::CPersistUtils::persist(MAPPED_ENCODING_FALLBACK_TAG, m_Fallback, inserter);
    core::CPersistUtils::persist(MAPPED_ENCODING_BINARY_TAG, m_Binary, inserter);
}

bool CDataFrameCategoryEncoder::CMappedEncoding::acceptRestoreTraverserForDerivedTypeState(
    core::CStateRestoreTraverser& traverser) {
    const std::string& name = traverser.name();
    RESTORE_NO_LOOP(MAPPED_ENCODING_MAP_TAG,
                    core::CPersistUtils::restore(MAPPED_ENCODING_MAP_TAG, m_Map, traverser))
    RESTORE_NO_LOOP(MAPPED_ENCODING_FALLBACK_TAG,
                    core::CPersistUtils::restore(MAPPED_ENCODING_FALLBACK_TAG,
                                                 m_Fallback, traverser))
    RESTORE_NO_LOOP(MAPPED_ENCODING_BINARY_TAG,
                    core::CPersistUtils::restore(MAPPED_ENCODING_BINARY_TAG, m_Binary, traverser))

    return true;
}

const std::string& CDataFrameCategoryEncoder::CMappedEncoding::typeString() const {
    return (m_Encoding == EEncoding::E_Frequency) ? FREQUENCY_ENCODING_TAG
                                                  : TARGET_MEAN_ENCODING_TAG;
}

const TDoubleVec& CDataFrameCategoryEncoder::CMappedEncoding::map() const {
    return m_Map;
}

double CDataFrameCategoryEncoder::CMappedEncoding::fallback() const {
    return m_Fallback;
}

CMakeDataFrameCategoryEncoder::CMakeDataFrameCategoryEncoder(std::size_t numberThreads,
                                                             const core::CDataFrame& frame,
                                                             std::size_t targetColumn)
    : m_NumberThreads{numberThreads}, m_Frame{&frame}, m_RowMask{frame.numberRows(), true},
      m_TargetColumn{targetColumn} {

    m_ColumnMask.resize(frame.numberColumns());
    std::iota(m_ColumnMask.begin(), m_ColumnMask.end(), 0);
    m_ColumnMask.erase(m_ColumnMask.begin() + targetColumn);
}

CMakeDataFrameCategoryEncoder&
CMakeDataFrameCategoryEncoder::minimumRowsPerFeature(std::size_t minimumRowsPerFeature) {
    m_MinimumRowsPerFeature = minimumRowsPerFeature;
    return *this;
}

CMakeDataFrameCategoryEncoder&
CMakeDataFrameCategoryEncoder::minimumFrequencyToOneHotEncode(TOptionalDouble minimumFrequencyToOneHotEncode) {
    if (minimumFrequencyToOneHotEncode != boost::none) {
        m_MinimumFrequencyToOneHotEncode = *minimumFrequencyToOneHotEncode;
    }
    return *this;
}

CMakeDataFrameCategoryEncoder&
CMakeDataFrameCategoryEncoder::redundancyWeight(TOptionalDouble redundancyWeight) {
    if (redundancyWeight != boost::none) {
        m_RedundancyWeight = *redundancyWeight;
    }
    return *this;
}

CMakeDataFrameCategoryEncoder&
CMakeDataFrameCategoryEncoder::minimumRelativeMicToSelectFeature(TOptionalDouble minimumRelativeMicToSelectFeature) {
    if (minimumRelativeMicToSelectFeature != boost::none) {
        m_MinimumRelativeMicToSelectFeature = *minimumRelativeMicToSelectFeature;
    }
    return *this;
}

CMakeDataFrameCategoryEncoder&
CMakeDataFrameCategoryEncoder::rowMask(core::CPackedBitVector rowMask) {
    m_RowMask = std::move(rowMask);
    return *this;
}

CMakeDataFrameCategoryEncoder& CMakeDataFrameCategoryEncoder::columnMask(TSizeVec columnMask) {
    m_ColumnMask = std::move(columnMask);
    return *this;
}

CMakeDataFrameCategoryEncoder::TEncodingUPtrVec CMakeDataFrameCategoryEncoder::makeEncodings() {

    TSizeVec metricColumnMask(m_ColumnMask);
    metricColumnMask.erase(
        std::remove_if(metricColumnMask.begin(), metricColumnMask.end(),
                       [this](std::size_t i) {
                           return i == m_TargetColumn ||
                                  m_Frame->columnIsCategorical()[i];
                       }),
        metricColumnMask.end());
    LOG_TRACE(<< "metric column mask = " << core::CContainerPrinter::print(metricColumnMask));

    TSizeVec categoricalColumnMask(m_ColumnMask);
    categoricalColumnMask.erase(
        std::remove_if(categoricalColumnMask.begin(), categoricalColumnMask.end(),
                       [this](std::size_t i) {
                           return i == m_TargetColumn ||
                                  m_Frame->columnIsCategorical()[i] == false;
                       }),
        categoricalColumnMask.end());
    LOG_TRACE(<< "categorical column mask = "
              << core::CContainerPrinter::print(categoricalColumnMask));

    // The top-level strategy is as follows:
    //
    // We one-hot encode the frequent categories with the highest non-zero MICe up
    // to the permitted overall feature count.
    // We mean value encode the remaining features where we have a representative
    // sample size.
    // We frequency encode all rare categories of a categorical feature.

    this->setupFrequencyEncoding(categoricalColumnMask);
    this->setupTargetMeanValueEncoding(categoricalColumnMask);
    this->finishEncoding(this->selectFeatures(metricColumnMask, categoricalColumnMask));

    return this->readEncodings();
}

CMakeDataFrameCategoryEncoder::TEncodingUPtrVec
CMakeDataFrameCategoryEncoder::readEncodings() const {

    // In the encoded row the layout of the encoding dimensions, for each categorical
    // feature, is as follows:
    //   (...| one-hot | mean target | frequency |...)
    //
    // The ones are in the order the categories appear in m_OneHotEncodedCategories.
    // For example, if m_OneHotEncodedCategories[feature] = (2, 5, 7) for any other
    // category the encoded row will contain (...| 0 0 0 | ...). For 2, 5 and 7 it
    // will contain (...| 1 0 0 |...), (...| 0 1 0 |...) and (...| 0 0 1 |...),
    // respectively.
    //
    // In the following we therefore 1) check to see if the category is being
    // one-hot encoded, 2) check if the encoding of the dimension, i.e. its offset
    // relative to the start of the encoding dimensions for the feature, is equal
    // to the position of the one for the category.

    TEncodingUPtrVec encodings;

    for (std::size_t encodedColumnIndex = 0;
         encodedColumnIndex < m_EncodedColumnInputColumnMap.size(); ++encodedColumnIndex) {

        std::size_t inputColumnIndex{m_EncodedColumnInputColumnMap[encodedColumnIndex]};
        double mic{m_EncodedColumnMics[encodedColumnIndex]};

        if (inputColumnIndex == m_TargetColumn ||
            m_Frame->columnIsCategorical()[inputColumnIndex] == false) {
            encodings.push_back(std::make_unique<CIdentityEncoding>(inputColumnIndex, mic));
            continue;
        }

        std::size_t categoryEncoding{m_EncodedColumnEncodingMap[encodedColumnIndex]};
        std::size_t numberOneHotCategories{
            m_OneHotEncodedCategories[inputColumnIndex].size()};

        if (categoryEncoding < numberOneHotCategories) {
            std::size_t hotCategory{m_OneHotEncodedCategories[inputColumnIndex][categoryEncoding]};
            encodings.push_back(std::make_unique<COneHotEncoding>(
                inputColumnIndex, mic, hotCategory));
            continue;
        }
        if (categoryEncoding == numberOneHotCategories &&
            m_InputColumnUsesFrequencyEncoding[inputColumnIndex]) {
            encodings.push_back(std::make_unique<CMappedEncoding>(
                inputColumnIndex, mic, E_Frequency, m_CategoryFrequencies[inputColumnIndex],
                m_MeanCategoryFrequencies[inputColumnIndex]));
            continue;
        }
        encodings.push_back(std::make_unique<CMappedEncoding>(
            inputColumnIndex, mic, E_TargetMean, m_CategoryTargetMeanValues[inputColumnIndex],
            m_MeanCategoryTargetMeanValues[inputColumnIndex]));
    }

    return encodings;
}

std::size_t CMakeDataFrameCategoryEncoder::encoding(std::size_t encodedColumnIndex) const {
    return m_EncodedColumnEncodingMap[encodedColumnIndex];
}

bool CMakeDataFrameCategoryEncoder::usesOneHotEncoding(std::size_t inputColumnIndex,
                                                       std::size_t category) const {
    return std::binary_search(m_OneHotEncodedCategories[inputColumnIndex].begin(),
                              m_OneHotEncodedCategories[inputColumnIndex].end(), category);
}

bool CMakeDataFrameCategoryEncoder::isRareCategory(std::size_t inputColumnIndex,
                                                   std::size_t category) const {
    return m_RareCategories[inputColumnIndex].find(category) !=
           m_RareCategories[inputColumnIndex].end();
}

CMakeDataFrameCategoryEncoder::TSizeDoublePrVecVec
CMakeDataFrameCategoryEncoder::mics(const CDataFrameUtils::CColumnValue& target,
                                    const TSizeVec& metricColumnMask,
                                    const TSizeVec& categoricalColumnMask) const {

    CDataFrameUtils::TEncoderFactoryVec encoderFactories(
        static_cast<std::size_t>(E_IdentityEncoding));
    encoderFactories[E_OneHot] = std::make_pair(
        [](std::size_t, std::size_t sampleColumn, std::size_t category) {
            return std::make_unique<CDataFrameUtils::COneHotCategoricalColumnValue>(
                sampleColumn, category);
        },
        m_MinimumFrequencyToOneHotEncode);
    encoderFactories[E_TargetMean] = std::make_pair(
        [this](std::size_t column, std::size_t sampleColumn, std::size_t) {
            return std::make_unique<CDataFrameUtils::CTargetMeanCategoricalColumnValue>(
                sampleColumn, m_RareCategories[column], m_CategoryTargetMeanValues[column]);
        },
        0.0);
    encoderFactories[E_Frequency] = std::make_pair(
        [this](std::size_t column, std::size_t sampleColumn, std::size_t) {
            return std::make_unique<CDataFrameUtils::CFrequencyCategoricalColumnValue>(
                sampleColumn, m_CategoryFrequencies[column]);
        },
        0.0);

    auto metricMics = CDataFrameUtils::metricMicWithColumn(target, *m_Frame, m_RowMask,
                                                           metricColumnMask);
    auto categoricalMics = CDataFrameUtils::categoricalMicWithColumn(
        target, m_NumberThreads, *m_Frame, m_RowMask, categoricalColumnMask, encoderFactories);

    TSizeDoublePrVecVec mics(std::move(categoricalMics[E_OneHot]));
    for (std::size_t i = 0; i < categoricalMics[E_TargetMean].size(); ++i) {
        if (categoricalMics[E_TargetMean][i].empty() == false) {
            mics[i].emplace_back(CATEGORY_FOR_TARGET_MEAN_ENCODING,
                                 categoricalMics[E_TargetMean][i][0].second);
        }
    }
    for (std::size_t i = 0; i < categoricalMics[E_Frequency].size(); ++i) {
        if (categoricalMics[E_Frequency][i].empty() == false) {
            mics[i].emplace_back(CATEGORY_FOR_FREQUENCY_ENCODING,
                                 categoricalMics[E_Frequency][i][0].second);
        }
    }
    for (std::size_t i = 0; i < metricMics.size(); ++i) {
        if (metricMics[i] > 0.0) {
            mics[i].emplace_back(CATEGORY_FOR_METRICS, metricMics[i]);
        }
    }
    LOG_TRACE(<< "MICe = " << core::CContainerPrinter::print(mics));

    return mics;
}

void CMakeDataFrameCategoryEncoder::setupFrequencyEncoding(const TSizeVec& categoricalColumnMask) {

    m_CategoryFrequencies = CDataFrameUtils::categoryFrequencies(
        m_NumberThreads, *m_Frame, m_RowMask, categoricalColumnMask);
    LOG_TRACE(<< "category frequencies = "
              << core::CContainerPrinter::print(m_CategoryFrequencies));

    m_MeanCategoryFrequencies.resize(m_CategoryFrequencies.size());
    m_RareCategories.resize(m_CategoryFrequencies.size());
    for (std::size_t i = 0; i < m_CategoryFrequencies.size(); ++i) {
        m_MeanCategoryFrequencies[i] =
            m_CategoryFrequencies[i].empty()
                ? 1.0
                : 1.0 / static_cast<double>(m_CategoryFrequencies[i].size());
        for (std::size_t j = 0; j < m_CategoryFrequencies[i].size(); ++j) {
            std::size_t count{static_cast<std::size_t>(
                m_CategoryFrequencies[i][j] * static_cast<double>(m_Frame->numberRows()) + 0.5)};
            if (count < m_MinimumRowsPerFeature) {
                m_RareCategories[i].insert(j);
            }
        }
    }
    LOG_TRACE(<< "mean category frequencies = "
              << core::CContainerPrinter::print(m_MeanCategoryFrequencies));
    LOG_TRACE(<< "rare categories = " << core::CContainerPrinter::print(m_RareCategories));
}

void CMakeDataFrameCategoryEncoder::setupTargetMeanValueEncoding(const TSizeVec& categoricalColumnMask) {

    m_CategoryTargetMeanValues = CDataFrameUtils::meanValueOfTargetForCategories(
        CDataFrameUtils::CMetricColumnValue{m_TargetColumn}, m_NumberThreads,
        *m_Frame, m_RowMask, categoricalColumnMask);
    LOG_TRACE(<< "category target mean values = "
              << core::CContainerPrinter::print(m_CategoryTargetMeanValues));

    m_MeanCategoryTargetMeanValues.resize(m_CategoryTargetMeanValues.size());
    for (std::size_t i = 0; i < m_CategoryTargetMeanValues.size(); ++i) {
        m_MeanCategoryTargetMeanValues[i] =
            m_CategoryTargetMeanValues[i].empty()
                ? 0.0
                : CBasicStatistics::mean(m_CategoryTargetMeanValues[i]);
    }
    LOG_TRACE(<< "mean category target mean values = "
              << core::CContainerPrinter::print(m_MeanCategoryTargetMeanValues));
}

CMakeDataFrameCategoryEncoder::TSizeSizePrDoubleMap
CMakeDataFrameCategoryEncoder::selectAllFeatures(const TSizeDoublePrVecVec& mics) {

    TSizeSizePrDoubleMap selectedFeatureMics;

    for (std::size_t feature = 0; feature < mics.size(); ++feature) {
        for (std::size_t i = 0; i < mics[feature].size(); ++i) {
            std::size_t category;
            double mic;
            std::tie(category, mic) = mics[feature][i];
            if (mic == 0.0) {
                continue;
            }
            LOG_TRACE(<< "Selected feature = " << feature << ", category = "
                      << print(category) << ", mic with target = " << mic);

            selectedFeatureMics[{feature, category}] = mic;

            if (isCategory(category)) {
                m_OneHotEncodedCategories[feature].push_back(category);
            } else if (isFrequency(category)) {
                m_InputColumnUsesFrequencyEncoding[feature] = true;
            } // else if (isTargetMean(category)) { nothing to do }
        }
    }

    LOG_TRACE(<< "one-hot encoded = "
              << core::CContainerPrinter::print(m_OneHotEncodedCategories));

    return selectedFeatureMics;
}

CMakeDataFrameCategoryEncoder::TSizeSizePrDoubleMap
CMakeDataFrameCategoryEncoder::selectFeatures(TSizeVec metricColumnMask,
                                              const TSizeVec& categoricalColumnMask) {

    // We want to choose features which provide independent information about the
    // target variable. Ideally, we'd recompute MICe w.r.t. target - f(x) with x
    // the features selected so far. This would be very computationally expensive
    // since it requires training a model f(.) on a subset of the features after
    // each decision. Instead, we use the average MICe between the unselected and
    // selected features as a useful proxy. This is essentially the mRMR approach
    // of Peng et al. albeit with MICe rather than mutual information. Except, it
    // also supports a redundancy weight, which should be non-negative and is used
    // to control the relative weight of MICe with the target vs the selected
    // variables. A value of zero means exclusively maximise MICe with the target
    // and as redundancy weight -> infinity means exclusively minimise MICe with
    // the selected variables.

    TSizeDoublePrVecVec mics(this->mics(CDataFrameUtils::CMetricColumnValue{m_TargetColumn},
                                        metricColumnMask, categoricalColumnMask));
    this->discardNuisanceFeatures(mics);
    LOG_TRACE(<< "features MICe = " << core::CContainerPrinter::print(mics));

    std::size_t numberAvailableFeatures{this->numberAvailableFeatures(mics)};
    std::size_t maximumNumberFeatures{
        (static_cast<std::size_t>(m_RowMask.manhattan()) + m_MinimumRowsPerFeature / 2) /
        m_MinimumRowsPerFeature};
    LOG_TRACE(<< "number possible features = " << numberAvailableFeatures
              << " maximum permitted features = " << maximumNumberFeatures);

    m_InputColumnUsesFrequencyEncoding.resize(m_Frame->numberColumns(), false);
    m_OneHotEncodedCategories.resize(m_Frame->numberColumns());

    TSizeSizePrDoubleMap selectedFeatureMics;

    if (maximumNumberFeatures >= numberAvailableFeatures) {

        selectedFeatureMics = this->selectAllFeatures(mics);

    } else {

        CMinRedundancyMaxRelevancyGreedySearch search{m_RedundancyWeight, mics};

        for (std::size_t i = 0; i < maximumNumberFeatures; ++i) {

            CFeatureRelevanceMinusRedundancy selected{search.selectNext()};

            double mic{selected.micWithDependentVariable()};
            std::size_t feature{selected.feature()};
            std::size_t category{selected.category()};
            LOG_TRACE(<< "Selected feature = " << feature << ", category = "
                      << print(category) << ", mic with target = " << mic);

            selectedFeatureMics[{feature, category}] = mic;

            if (selected.isCategory()) {
                m_OneHotEncodedCategories[feature].push_back(category);
            } else if (selected.isFrequency()) {
                m_InputColumnUsesFrequencyEncoding[feature] = true;
            } else if (selected.isMetric()) {
                metricColumnMask.erase(std::find(metricColumnMask.begin(),
                                                 metricColumnMask.end(), feature));
            } // else if (selected.isTargetMean()) { nothing to do }

            auto columnValue = selected.columnValue(
                m_RareCategories[feature], m_CategoryFrequencies[feature],
                m_CategoryTargetMeanValues[feature]);
            mics = this->mics(*columnValue, metricColumnMask, categoricalColumnMask);
            search.update(mics);
        }
    }

    for (auto& categories : m_OneHotEncodedCategories) {
        categories.shrink_to_fit();
        std::sort(categories.begin(), categories.end());
    }

    LOG_TRACE(<< "one-hot encoded = "
              << core::CContainerPrinter::print(m_OneHotEncodedCategories));
    LOG_TRACE(<< "selected features MICe = "
              << core::CContainerPrinter::print(selectedFeatureMics));

    return selectedFeatureMics;
}

void CMakeDataFrameCategoryEncoder::finishEncoding(TSizeSizePrDoubleMap selectedFeatureMics) {

    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;

    // Update the frequency and target mean encoding for one-hot and rare categories.

    for (std::size_t i = 0; i < m_OneHotEncodedCategories.size(); ++i) {
        TMeanAccumulator meanCategoryFrequency;
        TMeanAccumulator meanCategoryTargetMeanValue;
        for (auto category : m_OneHotEncodedCategories[i]) {
            double frequency{m_CategoryFrequencies[i][category]};
            double mean{m_CategoryTargetMeanValues[i][category]};
            meanCategoryFrequency.add(frequency, frequency);
            meanCategoryTargetMeanValue.add(mean, frequency);
        }
        for (auto category : m_OneHotEncodedCategories[i]) {
            m_CategoryFrequencies[i][category] = CBasicStatistics::mean(meanCategoryFrequency);
            m_CategoryTargetMeanValues[i][category] =
                CBasicStatistics::mean(meanCategoryTargetMeanValue);
        }
    }
    for (std::size_t i = 0; i < m_RareCategories.size(); ++i) {
        TMeanAccumulator meanCategoryTargetMeanValue;
        for (auto category : m_RareCategories[i]) {
            double frequency{m_CategoryFrequencies[i][category]};
            double mean{m_CategoryTargetMeanValues[i][category]};
            meanCategoryTargetMeanValue.add(mean, frequency);
        }
        for (auto category : m_RareCategories[i]) {
            m_CategoryTargetMeanValues[i][category] =
                CBasicStatistics::mean(meanCategoryTargetMeanValue);
        }
    }

    // Fill in a mapping from encoded column indices to raw column indices.

    selectedFeatureMics[{m_TargetColumn, CATEGORY_FOR_DEPENDENT_VARIABLE}] = 0.0;

    m_EncodedColumnMics.reserve(selectedFeatureMics.size());
    m_EncodedColumnInputColumnMap.reserve(selectedFeatureMics.size());
    m_EncodedColumnEncodingMap.reserve(selectedFeatureMics.size());

    auto i = selectedFeatureMics.begin();
    auto end = selectedFeatureMics.end();
    std::size_t encoding{0};
    for (;;) {
        std::size_t feature{i->first.first};
        double mic{i->second};
        m_EncodedColumnMics.push_back(mic);
        m_EncodedColumnInputColumnMap.push_back(feature);
        m_EncodedColumnEncodingMap.push_back(encoding);
        if (++i == end) {
            break;
        }
        encoding = i->first.first == feature ? encoding + 1 : 0;
    }

    LOG_TRACE(<< "feature vector MICe = "
              << core::CContainerPrinter::print(m_EncodedColumnMics));
    LOG_TRACE(<< "feature vector index to column map = "
              << core::CContainerPrinter::print(m_EncodedColumnInputColumnMap));
    LOG_TRACE(<< "feature vector index to encoding map = "
              << core::CContainerPrinter::print(m_EncodedColumnEncodingMap));
}

void CMakeDataFrameCategoryEncoder::discardNuisanceFeatures(TSizeDoublePrVecVec& mics) const {

    // Discard features carrying very little relative information about the target.
    // These will have a low chance of being selected and including them represents
    // a poor runtime QoR tradeoff. We achieve this by zeroing their MICe.

    using TSizeDoublePrVecItrVec = std::vector<TSizeDoublePrVec::iterator>;

    TSizeDoublePrVecItrVec flatMics;
    for (auto& featureMics : mics) {
        for (auto i = featureMics.begin(); i != featureMics.end(); ++i) {
            flatMics.push_back(i);
        }
    }
    std::stable_sort(flatMics.begin(), flatMics.end(),
                     [](auto lhs, auto rhs) { return lhs->second > rhs->second; });

    double totalMic{0.0};
    auto firstFeatureToDiscard =
        std::find_if(flatMics.begin(), flatMics.end(), [&](auto mic) {
            totalMic += mic->second;
            return mic->second < m_MinimumRelativeMicToSelectFeature * totalMic;
        });
    for (auto i = firstFeatureToDiscard; i != flatMics.end(); ++i) {
        (*i)->second = 0.0;
    }
}

std::size_t
CMakeDataFrameCategoryEncoder::numberAvailableFeatures(const TSizeDoublePrVecVec& mics) const {
    std::size_t count{0};
    for (const auto& featureMics : mics) {
        count += std::count_if(featureMics.begin(), featureMics.end(),
                               [](const auto& mic) { return mic.second > 0.0; });
    }
    return count;
}
}
}
