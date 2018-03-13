/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_model_CProbabilityCalibrator_h
#define INCLUDED_ml_model_CProbabilityCalibrator_h

#include <maths/ImportExport.h>

#include <boost/shared_ptr.hpp>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CQDigest;

//! \brief Calibrates a collection of probabilities.
//!
//! DESCRIPTION:\n
//! Calibrates a collection of probabilities s.t. they are close
//! to the historical empirical distribution of probabilities,
// i.e. we expect to see a probability of \f$p <= f\f$ approximately
//! \f$f * n\f$ given \f$n\f$ historical probabilities.
class MATHS_EXPORT CProbabilityCalibrator {
public:
    //! The type of calibration to perform:
    //!   -# Partial - only increase probabilities using the
    //!      historical fractions less the cutoff. Don't use
    //!      the fractions smaller than the cutoff instead
    //!      scale probabilities so the transform is continuous.
    //!   -# Full - perform a full calibration to historical
    //!      fractions.
    enum EStyle { E_PartialCalibration = 0, E_FullCalibration = 1 };

public:
    CProbabilityCalibrator(EStyle style, double cutoffProbability);

    //! \name Serialization
    //@{
    //! Persist state by passing information to the supplied inserter
    void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

    //! Create from an XML node tree.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);
    //@}

    //! Add \p probability.
    void add(double probability);

    //! Calibrate \p probability to the historic empirical
    //! distribution of probabilities.
    double calibrate(double probability) const;

private:
    using TQDigestPtr = boost::shared_ptr<CQDigest>;

private:
    //! The type of calibration to perform.
    EStyle m_Style;

    //! The smallest probability where we enforce a match
    //! with the historical fraction.
    double m_CutoffProbability;

    //! A summary of the historical probability quantiles.
    TQDigestPtr m_DiscreteProbabilityQuantiles;
};
}
}

#endif// INCLUDED_ml_model_CProbabilityCalibrator_h
