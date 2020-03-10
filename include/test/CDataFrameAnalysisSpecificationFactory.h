/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_test_CDataFrameAnalysisSpecificationFactory_h
#define INCLUDED_ml_test_CDataFrameAnalysisSpecificationFactory_h

#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>

#include <api/CDataFrameAnalysisSpecification.h>

#include <test/ImportExport.h>

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
    using TDataAdderUPtr = std::unique_ptr<core::CDataAdder>;
    using TPersisterSupplier = std::function<TDataAdderUPtr()>;
    using TDataSearcherUPtr = std::unique_ptr<core::CDataSearcher>;
    using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;
    using TSpecificationUPtr = std::unique_ptr<api::CDataFrameAnalysisSpecification>;

public:
    CDataFrameAnalysisSpecificationFactory();

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
    CDataFrameAnalysisSpecificationFactory&
    predictionCategoricalFieldNames(const TStrVec& categorical);
    CDataFrameAnalysisSpecificationFactory& predictionAlpha(double alpha);
    CDataFrameAnalysisSpecificationFactory& predictionLambda(double lambda);
    CDataFrameAnalysisSpecificationFactory& predictionGamma(double gamma);
    CDataFrameAnalysisSpecificationFactory& predictionSoftTreeDepthLimit(double limit);
    CDataFrameAnalysisSpecificationFactory& predictionSoftTreeDepthTolerance(double tolerance);
    CDataFrameAnalysisSpecificationFactory& predictionEta(double eta);
    CDataFrameAnalysisSpecificationFactory& predictionMaximumNumberTrees(std::size_t number);
    CDataFrameAnalysisSpecificationFactory& predictionFeatureBagFraction(double fraction);
    CDataFrameAnalysisSpecificationFactory& predictionNumberTopShapValues(std::size_t number);
    CDataFrameAnalysisSpecificationFactory&
    predictionPersisterSupplier(TPersisterSupplier* persisterSupplier);
    CDataFrameAnalysisSpecificationFactory&
    predictionRestoreSearcherSupplier(TRestoreSearcherSupplier* restoreSearcherSupplier);

    TSpecificationUPtr outlierSpec() const;
    TSpecificationUPtr predictionSpec(const std::string& analysis,
                                      const std::string& dependentVariable) const;

private:
    using TOptionalSize = boost::optional<std::size_t>;

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
    double m_Alpha = -1.0;
    double m_Lambda = -1.0;
    double m_Gamma = -1.0;
    double m_SoftTreeDepthLimit = -1.0;
    double m_SoftTreeDepthTolerance = -1.0;
    double m_Eta = -1.0;
    std::size_t m_MaximumNumberTrees = 0;
    double m_FeatureBagFraction = -1.0;
    std::size_t m_NumberTopShapValues = 0;
    TPersisterSupplier* m_PersisterSupplier = nullptr;
    TRestoreSearcherSupplier* m_RestoreSearcherSupplier = nullptr;
};
}
}

#endif //INCLUDED_ml_test_CDataFrameAnalysisSpecificationFactory_h
