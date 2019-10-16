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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace test {
//! \brief Collection of helping methods to create data frame analysis specifications for tests.
class CDataFrameAnalysisSpecificationFactory {
public:
    using TStrVec = std::vector<std::string>;
    using TDataAdderUPtr = std::unique_ptr<ml::core::CDataAdder>;
    using TPersisterSupplier = std::function<TDataAdderUPtr()>;
    using TDataSearcherUPtr = std::unique_ptr<ml::core::CDataSearcher>;
    using TRestoreSearcherSupplier = std::function<TDataSearcherUPtr()>;
    using TSpecificationUPtr = std::unique_ptr<api::CDataFrameAnalysisSpecification>;

public:
    static TSpecificationUPtr outlierSpec(std::size_t rows = 110,
                                          std::size_t cols = 5,
                                          std::size_t memoryLimit = 100000,
                                          const std::string& method = "",
                                          std::size_t numberNeighbours = 0,
                                          bool computeFeatureInfluence = false);

    static TSpecificationUPtr
    predictionSpec(const std::string& analysis,
                   const std::string& dependentVariable,
                   std::size_t rows = 100,
                   std::size_t cols = 5,
                   std::size_t memoryLimit = 3000000,
                   std::size_t numberRoundsPerHyperparameter = 0,
                   std::size_t bayesianOptimisationRestarts = 0,
                   const TStrVec& categoricalFieldNames = TStrVec{},
                   double alpha = -1.0,
                   double lambda = -1.0,
                   double gamma = -1.0,
                   double softTreeDepthLimit = -1.0,
                   double softTreeDepthTolerance = -1.0,
                   double eta = -1.0,
                   std::size_t maximumNumberTrees = 0,
                   double featureBagFraction = -1.0,
                   TPersisterSupplier* persisterSupplier = nullptr,
                   TRestoreSearcherSupplier* restoreSearcherSupplier = nullptr);

    static TSpecificationUPtr
    diskUsageTestSpec(std::size_t rows, std::size_t cols, bool diskUsageAllowed);
};
}
}

#endif //INCLUDED_ml_test_CDataFrameAnalysisSpecificationFactory_h
