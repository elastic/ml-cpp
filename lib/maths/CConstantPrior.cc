/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/CConstantPrior.h>

#include <core/CContainerPrinter.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/Constants.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CMathsFuncs.h>

#include <cmath>
#include <iomanip>
#include <ios>
#include <limits>
#include <sstream>

namespace ml {
namespace maths {

namespace {
using TOptionalDouble = boost::optional<double>;

//! Set the constant, validating the input.
void setConstant(double value, TOptionalDouble& result) {
    if (CMathsFuncs::isNan(value)) {
        LOG_ERROR(<< "NaN constant");
    } else {
        result.reset(value);
    }
}

// We use short field names to reduce the state size
const core::TPersistenceTag CONSTANT_TAG("a", "constant");

const std::string EMPTY_STRING;

const double LOG_TWO = std::log(2.0);
}

CConstantPrior::CConstantPrior(const TOptionalDouble& constant)
    : CPrior(maths_t::E_DiscreteData, 0.0) {
    if (constant) {
        setConstant(*constant, m_Constant);
    }
}

CConstantPrior::CConstantPrior(core::CStateRestoreTraverser& traverser)
    : CPrior(maths_t::E_DiscreteData, 0.0) {
    traverser.traverseSubLevel(std::bind(&CConstantPrior::acceptRestoreTraverser,
                                         this, std::placeholders::_1));
}

bool CConstantPrior::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        RESTORE_SETUP_TEARDOWN(CONSTANT_TAG, double constant,
                               core::CStringUtils::stringToType(traverser.value(), constant),
                               m_Constant.reset(constant))
    } while (traverser.next());
    return true;
}

CConstantPrior::EPrior CConstantPrior::type() const {
    return E_Constant;
}

CConstantPrior* CConstantPrior::clone() const {
    return new CConstantPrior(*this);
}

void CConstantPrior::setToNonInformative(double /*offset*/, double /*decayRate*/) {
    m_Constant.reset();
}

bool CConstantPrior::needsOffset() const {
    return false;
}

double CConstantPrior::adjustOffset(const TDouble1Vec& /*samples*/,
                                    const TDoubleWeightsAry1Vec& /*weights*/) {
    return 0.0;
}

double CConstantPrior::offset() const {
    return 0.0;
}

void CConstantPrior::addSamples(const TDouble1Vec& samples,
                                const TDoubleWeightsAry1Vec& /*weights*/) {
    if (m_Constant || samples.empty()) {
        return;
    }
    setConstant(samples[0], m_Constant);
}

void CConstantPrior::propagateForwardsByTime(double /*time*/) {
}

CConstantPrior::TDoubleDoublePr CConstantPrior::marginalLikelihoodSupport() const {
    return {boost::numeric::bounds<double>::lowest(),
            boost::numeric::bounds<double>::highest()};
}

double CConstantPrior::marginalLikelihoodMean() const {
    if (this->isNonInformative()) {
        return 0.0;
    }
    return *m_Constant;
}

double CConstantPrior::marginalLikelihoodMode(const TDoubleWeightsAry& /*weights*/) const {
    return this->marginalLikelihoodMean();
}

CConstantPrior::TDoubleDoublePr
CConstantPrior::marginalLikelihoodConfidenceInterval(double /*percentage*/,
                                                     const TDoubleWeightsAry& /*weights*/) const {
    if (this->isNonInformative()) {
        return this->marginalLikelihoodSupport();
    }
    return {*m_Constant, *m_Constant};
}

double CConstantPrior::marginalLikelihoodVariance(const TDoubleWeightsAry& /*weights*/) const {
    return this->isNonInformative() ? boost::numeric::bounds<double>::highest() : 0.0;
}

maths_t::EFloatingPointErrorStatus
CConstantPrior::jointLogMarginalLikelihood(const TDouble1Vec& samples,
                                           const TDoubleWeightsAry1Vec& weights,
                                           double& result) const {

    result = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute likelihood for empty sample set");
        return maths_t::E_FpFailed;
    }

    if (samples.size() != weights.size()) {
        LOG_ERROR(<< "Mismatch in samples '"
                  << core::CContainerPrinter::print(samples) << "' and weights '"
                  << core::CContainerPrinter::print(weights) << "'");
        return maths_t::E_FpFailed;
    }

    if (this->isNonInformative()) {
        // The non-informative likelihood is improper and effectively
        // zero everywhere. We use minus max double because
        // log(0) = HUGE_VALUE, which causes problems for Windows.
        // Calling code is notified when the calculation overflows
        // and should avoid taking the exponential since this will
        // underflow and pollute the floating point environment. This
        // may cause issues for some library function implementations
        // (see fe*exceptflag for more details).
        result = boost::numeric::bounds<double>::lowest();
        return maths_t::E_FpOverflowed;
    }

    double numberSamples = 0.0;

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i] != *m_Constant) {
            // Technically infinite, but just use minus max double.
            result = boost::numeric::bounds<double>::lowest();
            return maths_t::E_FpOverflowed;
        }

        numberSamples += maths_t::countForUpdate(weights[i]);
    }

    result = numberSamples * core::constants::LOG_MAX_DOUBLE;
    return maths_t::E_FpNoErrors;
}

void CConstantPrior::sampleMarginalLikelihood(std::size_t numberSamples,
                                              TDouble1Vec& samples) const {
    samples.clear();
    if (this->isNonInformative()) {
        return;
    }
    samples.resize(numberSamples, *m_Constant);
}

bool CConstantPrior::minusLogJointCdf(const TDouble1Vec& samples,
                                      const TDoubleWeightsAry1Vec& weights,
                                      double& lowerBound,
                                      double& upperBound) const {

    lowerBound = upperBound = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute c.d.f. for empty sample set");
        return false;
    }

    double numberSamples = 0.0;
    try {
        for (std::size_t i = 0; i < samples.size(); ++i) {
            numberSamples += maths_t::count(weights[i]);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute c.d.f. " << e.what());
        return false;
    }

    if (this->isNonInformative()) {
        // Note that -log(0.5) = log(2).
        lowerBound = upperBound = numberSamples * LOG_TWO;
        return true;
    }

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i] < *m_Constant) {
            lowerBound = upperBound = core::constants::LOG_MAX_DOUBLE;
            return true;
        }
    }

    // Note that log(1) = 0.
    lowerBound = upperBound = 0.0;

    return true;
}

bool CConstantPrior::minusLogJointCdfComplement(const TDouble1Vec& samples,
                                                const TDoubleWeightsAry1Vec& weights,
                                                double& lowerBound,
                                                double& upperBound) const {

    lowerBound = upperBound = 0.0;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute c.d.f. for empty sample set");
        return false;
    }

    double numberSamples = 0.0;
    try {
        for (std::size_t i = 0; i < samples.size(); ++i) {
            numberSamples += maths_t::count(weights[i]);
        }
    } catch (const std::exception& e) {
        LOG_ERROR(<< "Failed to compute c.d.f. " << e.what());
        return false;
    }

    if (this->isNonInformative()) {
        // Note that -log(0.5) = log(2).
        lowerBound = upperBound = numberSamples * LOG_TWO;
        return true;
    }

    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i] > *m_Constant) {
            lowerBound = upperBound = core::constants::LOG_MAX_DOUBLE;
            return true;
        }
    }

    // Note that log(1) = 0.
    lowerBound = upperBound = 0.0;

    return true;
}

bool CConstantPrior::probabilityOfLessLikelySamples(maths_t::EProbabilityCalculation /*calculation*/,
                                                    const TDouble1Vec& samples,
                                                    const TDoubleWeightsAry1Vec& /*weights*/,
                                                    double& lowerBound,
                                                    double& upperBound,
                                                    maths_t::ETail& tail) const {

    lowerBound = upperBound = 0.0;
    tail = maths_t::E_UndeterminedTail;

    if (samples.empty()) {
        LOG_ERROR(<< "Can't compute probability for empty sample set");
        return false;
    }

    lowerBound = upperBound = 1.0;

    if (this->isNonInformative()) {
        return true;
    }

    int tail_ = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i] != *m_Constant) {
            lowerBound = upperBound = 0.0;
        }
        if (samples[i] < *m_Constant) {
            tail_ = tail_ | maths_t::E_LeftTail;
        } else if (samples[i] > *m_Constant) {
            tail_ = tail_ | maths_t::E_RightTail;
        }
    }

    LOG_TRACE(<< "samples = " << core::CContainerPrinter::print(samples)
              << ", constant = " << *m_Constant << ", lowerBound = " << lowerBound
              << ", upperBound = " << upperBound << ", tail = " << tail);

    tail = static_cast<maths_t::ETail>(tail_);
    return true;
}

bool CConstantPrior::isNonInformative() const {
    return !m_Constant;
}

void CConstantPrior::print(const std::string& indent, std::string& result) const {
    result += core_t::LINE_ENDING + indent + "constant " +
              (this->isNonInformative() ? std::string("non-informative")
                                        : core::CStringUtils::typeToString(*m_Constant));
}

std::string CConstantPrior::printMarginalLikelihoodFunction(double /*weight*/) const {
    // The marginal likelihood is zero everywhere and infinity
    // at the constant so not particularly interesting and we don't
    // bother to define this function.
    return EMPTY_STRING;
}

std::string CConstantPrior::printJointDensityFunction() const {
    // The prior is (arguably) Dirichlet with infinite concentration
    // at the constant so not particularly interesting and we don't
    // bother to define this function.
    return EMPTY_STRING;
}

uint64_t CConstantPrior::checksum(uint64_t seed) const {
    seed = this->CPrior::checksum(seed);
    return CChecksum::calculate(seed, m_Constant);
}

void CConstantPrior::debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
    mem->setName("CConstantPrior");
}

std::size_t CConstantPrior::memoryUsage() const {
    return 0;
}

std::size_t CConstantPrior::staticSize() const {
    return sizeof(*this);
}

void CConstantPrior::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    if (m_Constant) {
        const std::string constantTag{CONSTANT_TAG};
        inserter.insertValue(constantTag, *m_Constant, core::CIEEE754::E_DoublePrecision);
    }
}

CConstantPrior::TOptionalDouble CConstantPrior::constant() const {
    return m_Constant;
}
}
}
