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

#include <maths/common/CMixtureDistribution.h>

#include <maths/common/CTools.h>

namespace ml {
namespace maths {
namespace common {
namespace {

using TDoubleDoublePr = std::pair<double, double>;

//! brief Invokes the support function on a distribution.
struct SSupport {
    template<typename DISTRIBUTION>
    TDoubleDoublePr operator()(const DISTRIBUTION& distribution) const {
        return support(distribution);
    }
};

//! \brief Invokes the mode function on a distribution.
struct SMode {
    template<typename DISTRIBUTION>
    double operator()(const DISTRIBUTION& distribution) const {
        return mode(distribution);
    }
};

//! \brief Invokes the mode function on a distribution.
struct SMean {
    template<typename DISTRIBUTION>
    double operator()(const DISTRIBUTION& distribution) const {
        return mean(distribution);
    }
};

//! \brief Invokes CTools::safePdf on a distribution.
struct SPdf {
    template<typename DISTRIBUTION>
    double operator()(const DISTRIBUTION& distribution, double x) const {
        return CTools::safePdf(distribution, x);
    }
};

//! \brief Invokes CTools::safeCdf on a distribution.
struct SCdf {
    template<typename DISTRIBUTION>
    double operator()(const DISTRIBUTION& distribution, double x) const {
        return CTools::safeCdf(distribution, x);
    }
};

//! \brief Invokes CTools::safeCdfComplement on a distribution.
struct SCdfComplement {
    template<typename DISTRIBUTION>
    double operator()(const DISTRIBUTION& distribution, double x) const {
        return CTools::safeCdfComplement(distribution, x);
    }
};

//! \brief Invokes the quantile function on a distribution.
struct SQuantile {
    template<typename DISTRIBUTION>
    double operator()(const DISTRIBUTION& distribution, double x) const {
        return quantile(distribution, x);
    }
};

//! \brief Invokes a specified binary action on a distribution.
template<typename RESULT, typename VISITOR_ACTION>
class CUnaryVisitor {
public:
    using result_type = RESULT;

public:
    template<typename DISTRIBUTION>
    RESULT operator()(const DISTRIBUTION& distribution) const {
        return action(distribution);
    }

private:
    VISITOR_ACTION action;
};

//! \brief Invokes a specified binary action on a distribution.
template<typename RESULT, typename VISITOR_ACTION>
class CBinaryVisitor {
public:
    using result_type = RESULT;

public:
    template<typename DISTRIBUTION>
    RESULT operator()(const DISTRIBUTION& distribution, double x) const {
        return action(distribution, x);
    }

private:
    VISITOR_ACTION action;
};
}

namespace mixture_detail {

CMixtureModeImpl::CMixtureModeImpl(const boost::math::normal_distribution<>& normal)
    : m_Distribution(normal) {
}

CMixtureModeImpl::CMixtureModeImpl(const boost::math::gamma_distribution<>& gamma)
    : m_Distribution(gamma) {
}

CMixtureModeImpl::CMixtureModeImpl(const boost::math::lognormal_distribution<>& lognormal)
    : m_Distribution(lognormal) {
}
}

CMixtureMode<false>::CMixtureMode(const boost::math::normal_distribution<>& normal)
    : mixture_detail::CMixtureModeImpl(normal) {
}

CMixtureMode<false>::CMixtureMode(const boost::math::gamma_distribution<>& gamma)
    : mixture_detail::CMixtureModeImpl(gamma) {
}

CMixtureMode<false>::CMixtureMode(const boost::math::lognormal_distribution<>& lognormal)
    : mixture_detail::CMixtureModeImpl(lognormal) {
}

CMixtureMode<true>::CMixtureMode(const CMixtureMode<false>& other)
    : mixture_detail::CMixtureModeImpl(other) {
}

mixture_detail::TDoubleDoublePr support(const CMixtureMode<false>& mode) {
    return mode.visit(CUnaryVisitor<TDoubleDoublePr, SSupport>());
}

double mode(const CMixtureMode<false>& mode) {
    return mode.visit(CUnaryVisitor<double, SMode>());
}

double mean(const CMixtureMode<false>& mode) {
    return mode.visit(CUnaryVisitor<double, SMean>());
}

double pdf(const CMixtureMode<false>& mode, double x) {
    return mode.visit(CBinaryVisitor<double, SPdf>(), x);
}

double cdf(const CMixtureMode<false>& mode, double x) {
    return mode.visit(CBinaryVisitor<double, SCdf>(), x);
}

double cdf(const CMixtureMode<true>& mode, double x) {
    return mode.visit(CBinaryVisitor<double, SCdfComplement>(), x);
}

double quantile(const CMixtureMode<false>& mode, double x) {
    return mode.visit(CBinaryVisitor<double, SQuantile>(), x);
}

CMixtureMode<true> complement(const CMixtureMode<false>& mode) {
    return CMixtureMode<true>(mode);
}
}
}
}
