/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_test_CDataFrameAnalysisSpecificationFactory_h
#define INCLUDED_ml_test_CDataFrameAnalysisSpecificationFactory_h

#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>

#include <maths/CBoostedTreeLoss.h>

#include <api/CDataFrameAnalysisSpecification.h>
#include <api/CDataFrameTrainBoostedTreeRegressionRunner.h>
#include <api/CDataFrameTrainBoostedTreeRunner.h>

#include <test/ImportExport.h>

#include <rapidjson/document.h>

#include <boost/optional.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace test {
//! \brief Collection of helping methods to create data frame analysis specifications for tests.
class TEST_EXPORT CDataFrameAnalysisSpecificationFactory {
public:
    using TStrVec = std::vector<std::string>;
    using TStrDoublePrVec = std::vector<std::pair<std::string, double>>;
    using TDataAdderUPtr = std::unique_ptr<core::CDataAdder>;
    using TPersisterSupplier = std::function<TDataAdderUPtr()>;
    using TDataSearcherUPtr = std::unique_ptr<core::CDataSearcher>;
    using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;
    using TSpecificationUPtr = std::unique_ptr<api::CDataFrameAnalysisSpecification>;
    using TLossFunctionType = maths::boosted_tree::ELossType;
    using TTask = api::CDataFrameTrainBoostedTreeRunner::ETask;

public:
    CDataFrameAnalysisSpecificationFactory();

    CDataFrameAnalysisSpecificationFactory(const CDataFrameAnalysisSpecificationFactory&) = delete;
    CDataFrameAnalysisSpecificationFactory&
    operator=(const CDataFrameAnalysisSpecificationFactory&) = delete;

    static const std::string& classification();
    static const std::string& regression();

    // Shared
    CDataFrameAnalysisSpecificationFactory& rows(std::size_t rows);
    CDataFrameAnalysisSpecificationFactory& columns(std::size_t columns);
    CDataFrameAnalysisSpecificationFactory& memoryLimit(std::size_t memoryLimit);
    CDataFrameAnalysisSpecificationFactory& missingString(const std::string& missing);
    CDataFrameAnalysisSpecificationFactory& diskUsageAllowed(bool disk);

    // Outliers
    CDataFrameAnalysisSpecificationFactory& outlierMethod(std::string method);
    CDataFrameAnalysisSpecificationFactory& outlierNumberNeighbours(std::size_t number);
    CDataFrameAnalysisSpecificationFactory& outlierComputeInfluence(bool compute);

    // Prediction
    CDataFrameAnalysisSpecificationFactory&
    predicitionNumberRoundsPerHyperparameter(std::size_t rounds);
    CDataFrameAnalysisSpecificationFactory&
    predictionBayesianOptimisationRestarts(std::size_t restarts);
    CDataFrameAnalysisSpecificationFactory& predictionFieldName(const std::string& name);
    CDataFrameAnalysisSpecificationFactory&
    predictionCategoricalFieldNames(const TStrVec& categorical);
    CDataFrameAnalysisSpecificationFactory& predictionAlpha(double alpha);
    CDataFrameAnalysisSpecificationFactory& predictionLambda(double lambda);
    CDataFrameAnalysisSpecificationFactory& predictionGamma(double gamma);
    CDataFrameAnalysisSpecificationFactory& predictionSoftTreeDepthLimit(double limit);
    CDataFrameAnalysisSpecificationFactory& predictionSoftTreeDepthTolerance(double tolerance);
    CDataFrameAnalysisSpecificationFactory& predictionEta(double eta);
    CDataFrameAnalysisSpecificationFactory& predictionMaximumNumberTrees(std::size_t number);
    CDataFrameAnalysisSpecificationFactory& predictionDownsampleFactor(double downsampleFactor);
    CDataFrameAnalysisSpecificationFactory& predictionFeatureBagFraction(double fraction);
    CDataFrameAnalysisSpecificationFactory& predictionNumberTopShapValues(std::size_t number);
    CDataFrameAnalysisSpecificationFactory&
    predictionCustomProcessor(const rapidjson::Value& value);
    CDataFrameAnalysisSpecificationFactory&
    predictionPersisterSupplier(TPersisterSupplier* persisterSupplier);
    CDataFrameAnalysisSpecificationFactory&
    predictionRestoreSearcherSupplier(TRestoreSearcherSupplier* restoreSearcherSupplier);
    CDataFrameAnalysisSpecificationFactory& earlyStoppingEnabled(bool earlyStoppingEnabled);
    CDataFrameAnalysisSpecificationFactory& task(TTask task);

    // Regression
    CDataFrameAnalysisSpecificationFactory& regressionLossFunction(TLossFunctionType lossFunction);
    CDataFrameAnalysisSpecificationFactory&
    regressionLossFunctionParameter(double lossFunctionParameter);

    // Classification
    CDataFrameAnalysisSpecificationFactory& numberClasses(std::size_t number);
    CDataFrameAnalysisSpecificationFactory& numberTopClasses(std::size_t number);
    CDataFrameAnalysisSpecificationFactory& predictionFieldType(const std::string& type);
    CDataFrameAnalysisSpecificationFactory&
    classificationWeights(const TStrDoublePrVec& weights);

    std::string outlierParams() const;
    TSpecificationUPtr outlierSpec() const;

    std::string predictionParams(const std::string& analysis,
                                 const std::string& dependentVariable) const;
    TSpecificationUPtr predictionSpec(const std::string& analysis,
                                      const std::string& dependentVariable) const;

private:
    using TOptionalSize = boost::optional<std::size_t>;
    using TOptionalDouble = boost::optional<double>;
    using TOptionalLossFunctionType = boost::optional<TLossFunctionType>;

private:
    // Shared
    TOptionalSize m_Rows;
    TOptionalSize m_Columns;
    TOptionalSize m_MemoryLimit;
    std::string m_MissingString;
    bool m_DiskUsageAllowed = true;
    // Outliers
    std::string m_Method;
    std::size_t m_NumberNeighbours = 0;
    bool m_ComputeFeatureInfluence = false;
    // Prediction
    std::size_t m_NumberRoundsPerHyperparameter = 0;
    std::size_t m_BayesianOptimisationRestarts = 0;
    TStrVec m_CategoricalFieldNames;
    std::string m_PredictionFieldName;
    double m_Alpha = -1.0;
    double m_Lambda = -1.0;
    double m_Gamma = -1.0;
    double m_SoftTreeDepthLimit = -1.0;
    double m_SoftTreeDepthTolerance = -1.0;
    double m_Eta = -1.0;
    std::size_t m_MaximumNumberTrees = 0;
    double m_DownsampleFactor = 0.0;
    double m_FeatureBagFraction = -1.0;
    std::size_t m_NumberTopShapValues = 0;
    TPersisterSupplier* m_PersisterSupplier = nullptr;
    TRestoreSearcherSupplier* m_RestoreSearcherSupplier = nullptr;
    rapidjson::Document m_CustomProcessors;
    TTask m_Task = TTask::E_Train;
    // Regression
    TOptionalLossFunctionType m_RegressionLossFunction;
    TOptionalDouble m_RegressionLossFunctionParameter;
    // Classification
    std::size_t m_NumberClasses = 2;
    std::size_t m_NumberTopClasses = 0;
    std::string m_PredictionFieldType;
    bool m_EarlyStoppingEnabled = true;
    TStrDoublePrVec m_ClassificationWeights;
};
}
}

#endif //INCLUDED_ml_test_CDataFrameAnalysisSpecificationFactory_h
