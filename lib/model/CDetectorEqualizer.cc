/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CDetectorEqualizer.h>

#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CTools.h>
#include <maths/Constants.h>

#include <boost/bind.hpp>
#include <boost/optional.hpp>

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
        inserter.insertLevel(SKETCH_TAG, boost::bind(&maths::CQuantileSketch::acceptPersistInserter, boost::cref(sketch.second), _1));
    }
}

bool CDetectorEqualizer::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    boost::optional<int> detector;
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(DETECTOR_TAG, detector.reset(0), core::CStringUtils::stringToType(traverser.value(), *detector),
                               /**/)
        if (name == SKETCH_TAG) {
            if (!detector) {
                LOG_ERROR("Expected the detector label first");
                return false;
            }
            m_Sketches.emplace_back(*detector, maths::CQuantileSketch(SKETCH_INTERPOLATION, SKETCH_SIZE));
            if (traverser.traverseSubLevel(
                    boost::bind(&maths::CQuantileSketch::acceptRestoreTraverser, boost::ref(m_Sketches.back().second), _1)) == false) {
                LOG_ERROR("Failed to restore SKETCH_TAG, got " << traverser.value());
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
    double logp = -maths::CTools::fastLog(probability);
    this->sketch(detector).add(logp);
}

double CDetectorEqualizer::correct(int detector, double probability) {
    LOG_TRACE("# detectors = " << m_Sketches.size());
    if (m_Sketches.size() == 1) {
        return probability;
    }

    const maths::CQuantileSketch& sketch = this->sketch(detector);

    for (const auto& sketch_ : m_Sketches) {
        if (sketch_.second.count() < MINIMUM_COUNT_FOR_CORRECTION) {
            return probability;
        }
    }

    static const double A = -maths::CTools::fastLog(maths::LARGEST_SIGNIFICANT_PROBABILITY);
    static const double B = -maths::CTools::fastLog(maths::SMALL_PROBABILITY);

    double logp = -maths::CTools::fastLog(probability);

    double percentage;
    if (sketch.cdf(logp, percentage)) {
        percentage *= 100.0;
        LOG_TRACE("log(p) = " << logp << ", c.d.f. = " << percentage);

        std::vector<double> logps;
        logps.reserve(m_Sketches.size());
        for (const auto& sketch_ : m_Sketches) {
            double logpi;
            if (sketch_.second.quantile(percentage, logpi)) {
                logps.push_back(logpi);
            }
        }
        std::sort(logps.begin(), logps.end());
        LOG_TRACE("quantiles = " << core::CContainerPrinter::print(logps));

        std::size_t n = logps.size();
        double logpc = n % 2 == 0 ? (logps[n / 2 - 1] + logps[n / 2]) / 2.0 : logps[n / 2];
        double alpha = maths::CTools::truncate((logp - A) / (B - A), 0.0, 1.0);
        LOG_TRACE("Corrected log(p) = " << -alpha * logpc - (1.0 - alpha) * logp);

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

uint64_t CDetectorEqualizer::checksum() const {
    return maths::CChecksum::calculate(0, m_Sketches);
}

double CDetectorEqualizer::largestProbabilityToCorrect() {
    return maths::LARGEST_SIGNIFICANT_PROBABILITY;
}

maths::CQuantileSketch& CDetectorEqualizer::sketch(int detector) {
    auto i = std::lower_bound(m_Sketches.begin(), m_Sketches.end(), detector, maths::COrderings::SFirstLess());
    if (i == m_Sketches.end() || i->first != detector) {
        i = m_Sketches.insert(i, {detector, maths::CQuantileSketch(SKETCH_INTERPOLATION, SKETCH_SIZE)});
    }
    return i->second;
}

const maths::CQuantileSketch::EInterpolation CDetectorEqualizer::SKETCH_INTERPOLATION(maths::CQuantileSketch::E_Linear);
const std::size_t CDetectorEqualizer::SKETCH_SIZE(100);
const double CDetectorEqualizer::MINIMUM_COUNT_FOR_CORRECTION(1.5);
}
}
