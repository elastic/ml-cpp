/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CDataClassifier_h
#define INCLUDED_ml_model_CDataClassifier_h

#include <core/CSmallVector.h>

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <string>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}

namespace model {

//! \brief Classifies a collection of values.
//!
//! DESCRIPTION:\n
//! Currently, this checks whether the values are all integers.
class MODEL_EXPORT CDataClassifier {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;

public:
    //! Update the classification with \p value.
    void add(model_t::EFeature feature, double value, unsigned int count);

    //! Update the classification with \p value.
    void add(model_t::EFeature feature, const TDouble1Vec& value, unsigned int count);

    //! Check if the values are all integers.
    bool isInteger() const;

    //! Check if the values are all positive.
    bool isNonNegative() const;

    // Consider adding function to check if the values live
    // on a lattice: i.e. x = {a + b*i} for integer i. This
    // would need to convert x(i) to integers and find the
    // g.c.d. of x(2) - x(1), x(3) - x(2) and so on.

    //! \name Persistence
    //@{
    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Create from part of an XML document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
    //@}

private:
    //! Set to false if the series contains non-integer values.
    bool m_IsInteger = true;

    //! Set to false if the series contains negative values.
    bool m_IsNonNegative = true;
};
}
}

#endif // INCLUDED_ml_model_CDataClassifier_h
