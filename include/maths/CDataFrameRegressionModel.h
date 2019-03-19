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
    using TProgressCallback = std::function<void(double)>;
    using TMemoryUsageCallback = std::function<void(std::uint64_t)>;

public:
    virtual ~CDataFrameRegressionModel() = default;

    //! Train the model on the values in \p frame.
    virtual void train(core::CDataFrame& frame,
                       TProgressCallback recordProgress = noop) = 0;

    //! Write the predictions of this model to \p frame.
    virtual void predict(core::CDataFrame& frame,
                         TProgressCallback recordProgress = noop) const = 0;

    //! Write this model to \p writer.
    virtual void write(core::CRapidJsonConcurrentLineWriter& writer) const = 0;

protected:
    static void noop(double);
};
}
}

#endif // INCLUDED_ml_maths_CDataFrameRegressionModel_h
