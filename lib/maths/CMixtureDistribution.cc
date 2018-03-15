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

#include <maths/CMixtureDistribution.h>

#include <maths/CTools.h>

namespace ml {
namespace maths {
namespace {

typedef std::pair<double, double> TDoubleDoublePr;

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
    typedef RESULT result_type;

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
    typedef RESULT result_type;

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
