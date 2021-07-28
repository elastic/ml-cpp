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

#ifndef INCLUDED_ml_maths_CLogTDistribution_h
#define INCLUDED_ml_maths_CLogTDistribution_h

#include <maths/ImportExport.h>

#include <boost/optional/optional_fwd.hpp>

#include <utility>

namespace ml {
namespace maths {

//! \brief Representation of a log t distribution.
//!
//! DESCRIPTION:\n
//! The log t distribution is the distribution of:
//! <pre class="fragment">
//!   \f$X = exp(s * Y + m)\f$
//! </pre>
//! where \f$Y\f$ is a random variable with the t distribution.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This follows the pattern used by boost::math::distributions
//! which defines lightweight objects to represent distributions
//! and free functions for computing various properties of the
//! distribution.
class MATHS_EXPORT CLogTDistribution {
public:
    using TDoubleDoublePr = std::pair<double, double>;
    using TOptionalDouble = boost::optional<double>;

public:
    CLogTDistribution(double degreesFreedom, double location, double scale);

    double degreesFreedom() const;
    double location() const;
    double scale() const;

private:
    double m_DegreesFreedom;
    double m_Location;
    double m_Scale;
};

//! Get the support for a log-t distribution.
MATHS_EXPORT
CLogTDistribution::TDoubleDoublePr support(const CLogTDistribution& distribution);

//! Compute the mode for \p distribution.
MATHS_EXPORT
double mode(const CLogTDistribution& distribution);

//! Get the finite local minimum if the distribution has one.
MATHS_EXPORT
CLogTDistribution::TOptionalDouble localMinimum(const CLogTDistribution& distribution);

//! Compute the p.d.f. at \p x for \p distribution.
MATHS_EXPORT
double pdf(const CLogTDistribution& distribution, double x);

//! Compute the c.d.f. at \p x for \p distribution.
MATHS_EXPORT
double cdf(const CLogTDistribution& distribution, double x);

//! Compute one minus the c.d.f. at \p x for \p distribution.
MATHS_EXPORT
double cdfComplement(const CLogTDistribution& distribution, double x);

//! Compute the \p q'th quantile for \p distribution.
MATHS_EXPORT
double quantile(const CLogTDistribution& distribution, double q);
}
}

#endif // INCLUDED_ml_maths_CLogTDistribution_h
