/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameRegressionModel_h
#define INCLUDED_ml_maths_CDataFrameRegressionModel_h

#include <core/CStatePersistInserter.h>

#include <maths/ImportExport.h>

#include <boost/optional.hpp>

#include <functional>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
class CRapidJsonConcurrentLineWriter;
}
namespace maths {

//! \brief Defines the interface for fitting and evaluating a regression model
//! on a data frame.
class MATHS_EXPORT CDataFrameRegressionModel {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TOptionalSizeVec = boost::optional<TSizeVec>;
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::uint64_t)>;
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;
    using TTrainingStateCallback = std::function<void(TPersistFunc)>;

public:
    virtual ~CDataFrameRegressionModel() = default;
    CDataFrameRegressionModel& operator=(const CDataFrameRegressionModel&) = delete;

    //! Train on the examples in the data frame supplied to the constructor.
    virtual void train() = 0;

    //! Write the predictions to the data frame supplied to the constructor.
    //!
    //! \warning This can only be called after train.
    virtual void predict() const = 0;

    //! Write SHAP values to the data frame supplied to the contructor.
    //!
    //! \warning This can only be called after train.
    virtual void computeShapValues() = 0;

    //! Get the feature weights the model has chosen.
    virtual const TDoubleVec& featureWeights() const = 0;

    //! Get the column containing the dependent variable.
    virtual std::size_t columnHoldingDependentVariable() const = 0;

    //! Get the column containing the model's prediction for the dependent variable.
    virtual std::size_t columnHoldingPrediction(std::size_t numberColumns) const = 0;

    //! Get the range of column indices with SHAP values
    //!
    //! \note The boost::irange can be sustituted by ranges::views::iota once this becomes available in C++20
    virtual const TOptionalSizeVec& columnsHoldingShapValues() const = 0;

public:
    static const std::string SHAP_PREFIX;

protected:
    CDataFrameRegressionModel(core::CDataFrame& frame,
                              TProgressCallback recordProgress,
                              TMemoryUsageCallback recordMemoryUsage,
                              TTrainingStateCallback recordTrainingState);
    core::CDataFrame& frame() const;
    const TProgressCallback& progressRecorder() const;
    const TMemoryUsageCallback& memoryUsageRecorder() const;
    const TTrainingStateCallback& trainingStateRecorder() const;

private:
    core::CDataFrame& m_Frame;
    TProgressCallback m_RecordProgress;
    TMemoryUsageCallback m_RecordMemoryUsage;
    TTrainingStateCallback m_RecordTrainingState;
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameRegressionModel_h
