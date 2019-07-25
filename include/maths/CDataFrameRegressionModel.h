/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDataFrameRegressionModel_h
#define INCLUDED_ml_maths_CDataFrameRegressionModel_h

#include <maths/ImportExport.h>

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
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::uint64_t)>;

public:
    virtual ~CDataFrameRegressionModel() = default;
    CDataFrameRegressionModel& operator=(const CDataFrameRegressionModel&) = delete;

    //! Train on the examples in the data frame supplied to the constructor.
    virtual void train(TProgressCallback recordProgress = noop) = 0;

    //! Write the predictions to the data frame supplied to the constructor.
    //!
    //! \warning This can only be called after train.
    virtual void predict(TProgressCallback recordProgress = noop) const = 0;

    //! Write this model to \p writer.
    //!
    //! \warning This can only be called after train.
    virtual void write(core::CRapidJsonConcurrentLineWriter& writer) const = 0;

    //! Get the feature weights the model has chosen.
    virtual TDoubleVec featureWeights() const = 0;

    //! Get the column containing the model's prediction for the dependent variable.
    virtual std::size_t columnHoldingPrediction(std::size_t numberColumns) const = 0;

protected:
    CDataFrameRegressionModel(core::CDataFrame& frame) : m_Frame{frame} {}
    core::CDataFrame& frame() const { return m_Frame; }
    static void noop(double);

private:
    core::CDataFrame& m_Frame;
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameRegressionModel_h
