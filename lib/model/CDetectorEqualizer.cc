/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <model/CDetectorEqualizer.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/common/CChecksum.h>
#include <maths/common/COrderings.h>
#include <maths/common/CTools.h>
#include <maths/common/Constants.h>

#include <optional>

namespace ml {
namespace model {
namespace {
const std::string DETECTOR_TAG("a");
const std::string SKETCH_TAG("b");
const std::string EMPTY_TAG("c");
}

void CDetectorEqualizer::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    if (m_Sketches.empty()) {
        inserter.insertValue(EMPTY_TAG, " ");
    }
    for (const auto& sketch : m_Sketches) {
        inserter.insertValue(DETECTOR_TAG, sketch.first);
        inserter.insertLevel(
            SKETCH_TAG, std::bind(&maths::common::CQuantileSketch::acceptPersistInserter,
                                  std::cref(sketch.second), std::placeholders::_1));
    }
}

bool CDetectorEqualizer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    std::optional<int> detector;
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(DETECTOR_TAG, detector.emplace(0),
                               core::CStringUtils::stringToType(traverser.value(), *detector),
                               /**/)
        if (name == SKETCH_TAG) {
            if (detector == std::nullopt) {
                LOG_ABORT(<< "Expected the detector label first");
            }
            m_Sketches.emplace_back(*detector, maths::common::CQuantileSketch(
                                                   SKETCH_INTERPOLATION, SKETCH_SIZE));
            if (traverser.traverseSubLevel(std::bind(
                    &maths::common::CQuantileSketch::acceptRestoreTraverser,
                    std::ref(m_Sketches.back().second), std::placeholders::_1)) == false) {
                LOG_ERROR(<< "Failed to restore SKETCH_TAG, got " << traverser.value());
                m_Sketches.pop_back();
                return false;
            }
            detector.reset();
            continue;
        }
    } while (traverser.next());
    return true;
}

void CDetectorEqualizer::add(int detector, double probability) {
    double logp = -maths::common::CTools::fastLog(probability);
    this->sketch(detector).add(logp);
}

double CDetectorEqualizer::correct(int detector, double probability) {
    LOG_TRACE(<< "# detectors = " << m_Sketches.size());
    if (m_Sketches.size() == 1) {
        return probability;
    }

    const maths::common::CQuantileSketch& sketch = this->sketch(detector);

    for (const auto& sketch_ : m_Sketches) {
        if (sketch_.second.count() < MINIMUM_COUNT_FOR_CORRECTION) {
            return probability;
        }
    }

    static const double A = -maths::common::CTools::fastLog(
        maths::common::LARGEST_SIGNIFICANT_PROBABILITY);
    static const double B = -maths::common::CTools::fastLog(maths::common::SMALL_PROBABILITY);

    double logp = -maths::common::CTools::fastLog(probability);

    double percentage;
    if (sketch.cdf(logp, percentage)) {
        percentage *= 100.0;
        LOG_TRACE(<< "log(p) = " << logp << ", c.d.f. = " << percentage);

        std::vector<double> logps;
        logps.reserve(m_Sketches.size());
        for (const auto& sketch_ : m_Sketches) {
            double logpi;
            if (sketch_.second.quantile(percentage, logpi)) {
                logps.push_back(logpi);
            }
        }
        std::sort(logps.begin(), logps.end());
        LOG_TRACE(<< "quantiles = " << logps);

        std::size_t n = logps.size();
        double logpc = n % 2 == 0 ? (logps[n / 2 - 1] + logps[n / 2]) / 2.0
                                  : logps[n / 2];
        double alpha = maths::common::CTools::truncate((logp - A) / (B - A), 0.0, 1.0);
        LOG_TRACE(<< "Corrected log(p) = " << -alpha * logpc - (1.0 - alpha) * logp);

        return std::exp(-alpha * logpc - (1.0 - alpha) * logp);
    }

    return probability;
}

void CDetectorEqualizer::clear() {
    m_Sketches.clear();
}

void CDetectorEqualizer::age(double factor) {
    for (auto& sketch : m_Sketches) {
        sketch.second.age(factor);
    }
}

std::uint64_t CDetectorEqualizer::checksum() const {
    return maths::common::CChecksum::calculate(0, m_Sketches);
}

double CDetectorEqualizer::largestProbabilityToCorrect() {
    return maths::common::LARGEST_SIGNIFICANT_PROBABILITY;
}

maths::common::CQuantileSketch& CDetectorEqualizer::sketch(int detector) {
    auto i = std::lower_bound(m_Sketches.begin(), m_Sketches.end(), detector,
                              maths::common::COrderings::SFirstLess());
    if (i == m_Sketches.end() || i->first != detector) {
        i = m_Sketches.insert(i, {detector, maths::common::CQuantileSketch(
                                                SKETCH_INTERPOLATION, SKETCH_SIZE)});
    }
    return i->second;
}

const maths::common::CQuantileSketch::EInterpolation
    CDetectorEqualizer::SKETCH_INTERPOLATION(maths::common::CQuantileSketch::E_Linear);
const std::size_t CDetectorEqualizer::SKETCH_SIZE(100);
const double CDetectorEqualizer::MINIMUM_COUNT_FOR_CORRECTION(1.5);
}
}
