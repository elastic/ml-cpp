/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_api_CDataFrameBoostedTreeRunner_h
#define INCLUDED_ml_api_CDataFrameBoostedTreeRunner_h

#include <core/CDataSearcher.h>

#include <api/CDataFrameAnalysisRunner.h>
#include <api/CDataFrameAnalysisSpecification.h>
#include <api/ImportExport.h>

#include <rapidjson/fwd.h>

#include <atomic>

namespace ml {
namespace maths {
class CBoostedTree;
class CBoostedTreeFactory;
}
namespace api {

//! \brief Runs boosted tree regression on a core::CDataFrame.
class API_EXPORT CDataFrameBoostedTreeRunner final : public CDataFrameAnalysisRunner {
public:
    //! This is not intended to be called directly: use CDataFrameBoostedTreeRunnerFactory.
    CDataFrameBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec,
                                const rapidjson::Value& jsonParameters);

    //! This is not intended to be called directly: use CDataFrameBoostedTreeRunnerFactory.
    CDataFrameBoostedTreeRunner(const CDataFrameAnalysisSpecification& spec);

    ~CDataFrameBoostedTreeRunner() override;

    //! \return The number of columns this adds to the data frame.
    std::size_t numberExtraColumns() const override;

    //! Write the prediction for \p row to \p writer.
    void writeOneRow(const TStrVec& featureNames,
                     TRowRef row,
                     core::CRapidJsonConcurrentLineWriter& writer) const override;

    void serializeRunner(const TStrVec& fieldNames,
                         core::CRapidJsonConcurrentLineWriter& writer) const override;

private:
    using TBoostedTreeUPtr = std::unique_ptr<maths::CBoostedTree>;
    using TBoostedTreeFactoryUPtr = std::unique_ptr<maths::CBoostedTreeFactory>;
    using TDataSearcherUPtr = CDataFrameAnalysisSpecification::TDataSearcherUPtr;
    using TMemoryEstimator = std::function<void(std::int64_t)>;

private:
    void runImpl(const TStrVec& featureNames, core::CDataFrame& frame) override;
    std::size_t estimateBookkeepingMemoryUsage(std::size_t numberPartitions,
                                               std::size_t totalNumberRows,
                                               std::size_t partitionNumberRows,
                                               std::size_t numberColumns) const override;
    TMemoryEstimator memoryEstimator();

    bool restoreBoostedTree(core::CDataFrame& frame,
                            std::size_t dependentVariableColumn,
                            TDataSearcherUPtr& restoreSearcher);

private:
    // Note custom config is written directly to the factory object.

    std::string m_DependentVariableFieldName;
    std::string m_PredictionFieldName;
    TBoostedTreeFactoryUPtr m_BoostedTreeFactory;
    TBoostedTreeUPtr m_BoostedTree;
    std::atomic<std::int64_t> m_Memory;
};

//! \brief Makes a core::CDataFrame boosted tree regression runner.
class API_EXPORT CDataFrameBoostedTreeRunnerFactory final : public CDataFrameAnalysisRunnerFactory {
public:
    const std::string& name() const override;

private:
    static const std::string NAME;

private:
    TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec) const override;
    TRunnerUPtr makeImpl(const CDataFrameAnalysisSpecification& spec,
                         const rapidjson::Value& params) const override;
};
}
}

#endif // INCLUDED_ml_api_CDataFrameBoostedTreeRunner_h
