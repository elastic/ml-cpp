/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFramePredictiveModel_h
#define INCLUDED_ml_maths_CDataFramePredictiveModel_h

#include <core/CStatePersistInserter.h>

#include <maths/ImportExport.h>

#include <boost/optional.hpp>
#include <boost/range/irange.hpp>

#include <functional>
#include <utility>
#include <vector>

namespace ml {
namespace core {
class CDataFrame;
class CRapidJsonConcurrentLineWriter;
}
namespace maths {

//! \brief Defines the interface for fitting and inferring a predictive model
//! with a data frame.
class MATHS_EXPORT CDataFramePredictiveModel {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeRange = boost::integer_range<std::size_t>;
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::int64_t)>;
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;
    using TTrainingStateCallback = std::function<void(TPersistFunc)>;

    //! The objective for the classification decision (given predicted class probabilities).
    enum EClassAssignmentObjective {
        E_Accuracy,     //!< Maximize prediction accuracy.
        E_MinimumRecall //!< Maximize the minimum per class recall.
    };

public:
    static const std::string SHAP_PREFIX;

public:
    virtual ~CDataFramePredictiveModel() = default;
    CDataFramePredictiveModel& operator=(const CDataFramePredictiveModel&) = delete;

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

    //! Get the column containing the dependent variable.
    virtual std::size_t columnHoldingDependentVariable() const = 0;

    //! Get the column containing the model's prediction for the dependent variable.
    virtual std::size_t columnHoldingPrediction() const = 0;

    //! Get the number of largest SHAP values that will be returned for every row.
    virtual std::size_t topShapValues() const = 0;

    //! Get the optional vector of column indices with SHAP values
    virtual TSizeRange columnsHoldingShapValues() const = 0;

    //! Get the probability threshold at which to classify a row as class one.
    virtual double probabilityAtWhichToAssignClassOne() const = 0;

    //! \name Test Only
    //@{
    //! Get the weight that has been chosen for each feature for training.
    virtual const TDoubleVec& featureWeightsForTraining() const = 0;
    //@}

protected:
    CDataFramePredictiveModel(core::CDataFrame& frame,
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

#endif // INCLUDED_ml_maths_CDataFramePredictiveModel_h
