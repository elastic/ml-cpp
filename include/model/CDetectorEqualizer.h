/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CDetectorEqualizer_h
#define INCLUDED_ml_model_CDetectorEqualizer_h

#include <maths/CQuantileSketch.h>

#include <model/ImportExport.h>

#include <cstddef>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
class CLimits;
class CModelConfig;

//! \brief Equalizers a set of detector probabilities at a result node.
//!
//! DESCRIPTION:\n
//! This maintains a quantile sketch of a set of minus log probabilities
//! for each detector. A corrected probability is obtained by converting
//! raw probabilities to a rank and then reading off median probability
//! for that rank over all detectors.
class MODEL_EXPORT CDetectorEqualizer {
public:
    using TIntQuantileSketchPr = std::pair<int, maths::CQuantileSketch>;
    using TIntQuantileSketchPrVec = std::vector<TIntQuantileSketchPr>;

public:
    //! Add \p probability to the detector's quantile sketch.
    void add(int detector, double probability);

    //! Correct \p probability to account for detector differences.
    double correct(int detector, double probability);

    //! Clear all sketches.
    void clear();

    //! Age the sketches by reducing the count.
    void age(double factor);

    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Restore reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Get a checksum for the equalizer.
    uint64_t checksum() const;

    //! Get the largest probability that will be corrected.
    static double largestProbabilityToCorrect();

private:
    //! Get the sketch for \p detector.
    maths::CQuantileSketch& sketch(int detector);

private:
    //! The style of interpolation to use for the sketch.
    static const maths::CQuantileSketch::EInterpolation SKETCH_INTERPOLATION;
    //! The maximum size of the quantile sketch.
    static const std::size_t SKETCH_SIZE;
    //! The minimum count in a detector's sketch for which we'll
    //! apply a correction to the probability.
    static const double MINIMUM_COUNT_FOR_CORRECTION;

private:
    //! The sketches (one for each detector).
    TIntQuantileSketchPrVec m_Sketches;
};
}
}

#endif // INCLUDED_ml_model_CDetectorEqualizer_h
