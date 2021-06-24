/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CEntropySketch.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CPRNG.h>
#include <maths/CSampling.h>

#include <boost/math/constants/constants.hpp>

#include <cmath>

namespace ml {
namespace maths {

CEntropySketch::CEntropySketch(std::size_t k) : m_Y(0), m_Yi(k, 0.0) {
}

void CEntropySketch::add(std::size_t category, uint64_t count) {
    m_Y += count;
    TDoubleVec projection;
    this->generateProjection(category, projection);
    for (std::size_t i = 0; i < projection.size(); ++i) {
        m_Yi[i] += projection[i] * static_cast<double>(count);
    }
}

double CEntropySketch::calculate() const {
    double h = 0.0;
    for (std::size_t i = 0; i < m_Yi.size(); ++i) {
        h += std::exp(m_Yi[i] / static_cast<double>(m_Y));
    }
    return -std::log(h / static_cast<double>(m_Yi.size()));
}

void CEntropySketch::generateProjection(std::size_t category, TDoubleVec& projection) {
    CPRNG::CXorOShiro128Plus rng(category);
    CSampling::uniformSample(rng, 0.0, 1.0, 2 * m_Yi.size(), projection);
    for (std::size_t i = 0; i < projection.size(); i += 2) {
        double w1 = boost::math::double_constants::pi * (projection[i] - 0.5);
        double w2 = -std::log(projection[i + 1]);
        projection[i / 2] =
            std::tan(w1) * (boost::math::double_constants::half_pi - w1) +
            std::log(w2 * std::cos(w1) / (boost::math::double_constants::half_pi - w1));
    }
    projection.resize(m_Yi.size());
    LOG_TRACE(<< "projection = " << core::CContainerPrinter::print(projection));
}
}
}
