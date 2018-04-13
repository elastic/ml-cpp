/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <config/CDataSemantics.h>

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <maths/CMixtureDistribution.h>
#include <maths/CNaturalBreaksClassifier.h>

#include <boost/math/distributions/normal.hpp>

#include <cmath>

namespace ml {
namespace config {
namespace {
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;

//! \brief Wraps up a mixture model.
//!
//! DESCRIPTION:\n
//! Wraps up the functionality to cluster a set of data and generate
//! a mixture model from the clustering. This is used to test for
//! numeric categorical fields by testing the BIC of a mixture model
//! verses a categorical model for the data.
class CMixtureData {
public:
    CMixtureData(double count, std::size_t N) : m_Count(count), m_Classifier(N) {}

    //! Add the data point \p xi with count \p ni.
    void add(double xi, double ni) { m_Classifier.add(xi, ni); }

    //! Compute the scale for a mixture of \p m.
    double scale(std::size_t m) {
        using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

        TSizeVec split;
        m_Classifier.naturalBreaks(m, 2, split);
        split.insert(split.begin(), 0);
        m_Classifier.categories(m, 2, m_Categories);
        TSizeVec counts;
        counts.reserve(m);
        for (std::size_t i = 1u; i < split.size(); ++i) {
            counts.push_back(split[i] - split[i - 1]);
        }

        TMeanAccumulator scale;
        for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
            double ci = maths::CBasicStatistics::count(m_Categories[i]);
            double vi = maths::CBasicStatistics::maximumLikelihoodVariance(m_Categories[i]);
            double si = std::max(3.0 * std::sqrt(vi), 1.0 / boost::math::constants::root_two_pi<double>());
            scale.add(static_cast<double>(counts[i]) / si, ci);
        }
        return maths::CBasicStatistics::mean(scale);
    }

    //! Populate a mixture of \p m.
    void populate(std::size_t m) {
        this->clear();

        TSizeVec split;
        m_Classifier.naturalBreaks(m, 2, split);
        split.insert(split.begin(), 0);
        m_Classifier.categories(m, 2, m_Categories);
        TSizeVec counts;
        counts.reserve(m);
        for (std::size_t i = 1u; i < split.size(); ++i) {
            counts.push_back(split[i] - split[i - 1]);
        }
        LOG_TRACE("m_Categories = " << core::CContainerPrinter::print(m_Categories));

        for (std::size_t i = 0u; i < m_Categories.size(); ++i) {
            double ci = maths::CBasicStatistics::count(m_Categories[i]);
            double mi = maths::CBasicStatistics::mean(m_Categories[i]);
            double vi = maths::CBasicStatistics::maximumLikelihoodVariance(m_Categories[i]);
            double si = std::max(std::sqrt(vi), 1.0 / boost::math::constants::root_two_pi<double>());
            m_Gmm.weights().push_back(ci / m_Count);
            m_Gmm.modes().push_back(boost::math::normal_distribution<>(mi, si));
        }
        LOG_TRACE("GMM = '" << m_Gmm.print() << "'");
    }

    //! Get the number of parameters in the mixture.
    double parameters() const { return 3.0 * static_cast<double>(m_Categories.size()) - 1.0; }

    //! Compute the value of the density function at \p x.
    double pdf(double x) const { return maths::pdf(m_Gmm, x); }

private:
    using TNormalVec = std::vector<boost::math::normal_distribution<>>;
    using TGMM = maths::CMixtureDistribution<boost::math::normal_distribution<>>;

private:
    void clear() {
        m_Categories.clear();
        m_Gmm.weights().clear();
        m_Gmm.modes().clear();
    }

private:
    double m_Count;
    maths::CNaturalBreaksClassifier m_Classifier;
    maths::CNaturalBreaksClassifier::TTupleVec m_Categories;
    TGMM m_Gmm;
};
}

CDataSemantics::CDataSemantics(TOptionalUserDataType override)
    : m_Type(config_t::E_UndeterminedType),
      m_Override(override),
      m_Count(0.0),
      m_NumericProportion(0.0),
      m_IntegerProportion(0.0),
      m_EmpiricalDistributionOverflowed(false) {
}

void CDataSemantics::add(const std::string& example) {
    m_Count += 1.0;

    maths::COrdinal value;

    std::string trimmed = example;
    core::CStringUtils::trimWhitespace(trimmed);
    int64_t asInt64;
    uint64_t asUInt64;
    double asDouble;
    if (core::CStringUtils::stringToTypeSilent(trimmed, asInt64)) {
        value = this->addInteger(asInt64);
    } else if (core::CStringUtils::stringToTypeSilent(trimmed, asUInt64)) {
        value = this->addPositiveInteger(asUInt64);
    } else if (core::CStringUtils::stringToTypeSilent(trimmed, asDouble)) {
        value = this->addReal(asDouble);
    }

    if (!value.isNan()) {
        m_Smallest.add(value);
        m_Largest.add(value);
    } else if (m_NonNumericValues.size() < 2 &&
               std::find(m_NonNumericValues.begin(), m_NonNumericValues.end(), trimmed) == m_NonNumericValues.end()) {
        m_NonNumericValues.push_back(trimmed);
    }

    if (m_DistinctValues.size() < 3 && std::find(m_DistinctValues.begin(), m_DistinctValues.end(), example) == m_DistinctValues.end()) {
        m_DistinctValues.push_back(example);
    }

    if (!m_EmpiricalDistributionOverflowed && !value.isNan()) {
        ++m_EmpiricalDistribution[value];
        if (m_EmpiricalDistribution.size() > MAXIMUM_EMPIRICAL_DISTRIBUTION_SIZE) {
            m_EmpiricalDistributionOverflowed = true;
            TOrdinalSizeUMap empty;
            m_EmpiricalDistribution.swap(empty);
        }
    }
}

void CDataSemantics::computeType() {
    if (m_Override) {
        switch (*m_Override) {
        case config_t::E_UserCategorical:
            m_Type = this->categoricalType();
            return;
        case config_t::E_UserNumeric:
            m_Type = this->isInteger() ? this->integerType() : this->realType();
            return;
        }
    }

    LOG_TRACE("count = " << m_Count);
    if (m_Count == 0.0) {
        m_Type = config_t::E_UndeterminedType;
        return;
    }

    if (m_DistinctValues.size() == 2) {
        m_Type = config_t::E_Binary;
        return;
    }

    LOG_TRACE("numeric proportion = " << m_NumericProportion);
    if (!this->isNumeric() || !this->GMMGoodFit()) {
        m_Type = config_t::E_Categorical;
        return;
    }

    LOG_TRACE("integer proportion = " << m_IntegerProportion);
    m_Type = this->isInteger() ? this->integerType() : this->realType();
}

config_t::EDataType CDataSemantics::type() const {
    return m_Type;
}

config_t::EDataType CDataSemantics::categoricalType() const {
    return m_DistinctValues.size() == 2 ? config_t::E_Binary : config_t::E_Categorical;
}

config_t::EDataType CDataSemantics::realType() const {
    return m_Smallest[0] < maths::COrdinal(0.0) ? config_t::E_Real : config_t::E_PositiveReal;
}

config_t::EDataType CDataSemantics::integerType() const {
    return m_Smallest[0] < maths::COrdinal(uint64_t(0)) ? config_t::E_Integer : config_t::E_PositiveInteger;
}

bool CDataSemantics::isNumeric() const {
    return m_NumericProportion >= NUMERIC_PROPORTION_FOR_METRIC_STRICT ||
           (m_NonNumericValues.size() < 2 && m_NumericProportion >= NUMERIC_PROPORTION_FOR_METRIC_WITH_SUSPECTED_MISSING_VALUES);
}

bool CDataSemantics::isInteger() const {
    return m_IntegerProportion / m_NumericProportion >= INTEGER_PRORORTION_FOR_INTEGER;
}

bool CDataSemantics::GMMGoodFit() const {
    if (m_EmpiricalDistributionOverflowed) {
        return true;
    }

    // The idea is to check the goodness-of-fit of a categorical model
    // to the data verses a normal mixture.

    using TOrdinalSizeUMapCItr = TOrdinalSizeUMap::const_iterator;

    std::size_t N = m_EmpiricalDistribution.size();
    LOG_TRACE("N = " << N);

    double logc = std::log(m_Count);
    double smallest = m_Smallest[0].asDouble();
    double offset = std::max(-smallest + 1.0, 0.0);
    LOG_TRACE("offset = " << offset);

    double categoricalBIC = static_cast<double>(N - 1) * logc;
    for (TOrdinalSizeUMapCItr i = m_EmpiricalDistribution.begin(); i != m_EmpiricalDistribution.end(); ++i) {
        double ni = static_cast<double>(i->second);
        categoricalBIC -= 2.0 * ni * std::log(ni / m_Count);
    }
    LOG_TRACE("categorical BIC = " << categoricalBIC);

    std::size_t M = std::min(m_EmpiricalDistribution.size() / 2, std::size_t(100));
    LOG_TRACE("m = " << M);

    for (std::size_t m = 1u; m <= M; ++m) {
        double scale = 1.0;
        {
            CMixtureData scaling(m_Count, N);
            for (TOrdinalSizeUMapCItr i = m_EmpiricalDistribution.begin(); i != m_EmpiricalDistribution.end(); ++i) {
                double xi = i->first.asDouble();
                double ni = static_cast<double>(i->second);
                scaling.add(xi, ni);
            }
            scale = std::min(scaling.scale(m), 1.0);
        }
        LOG_TRACE("scale = " << scale);

        CMixtureData light(m_Count, N);
        CMixtureData heavy(m_Count, N);
        for (TOrdinalSizeUMapCItr i = m_EmpiricalDistribution.begin(); i != m_EmpiricalDistribution.end(); ++i) {
            double xi = smallest + scale * (i->first.asDouble() - smallest);
            double ni = static_cast<double>(i->second);
            light.add(xi, ni);
            heavy.add(std::log(xi + offset), ni);
        }

        try {
            light.populate(m);
            heavy.populate(m);

            double lightGmmBIC = light.parameters() * logc;
            double heavyGmmBIC = heavy.parameters() * logc;
            for (TOrdinalSizeUMapCItr i = m_EmpiricalDistribution.begin(); i != m_EmpiricalDistribution.end(); ++i) {
                double xi = smallest + scale * (i->first.asDouble() - smallest);
                double ni = static_cast<double>(i->second);
                double fx = light.pdf(xi);
                double gx = 1.0 / (xi + offset) * heavy.pdf(std::log(xi + offset));
                lightGmmBIC -= 2.0 * ni * (fx == 0.0 ? boost::numeric::bounds<double>::lowest() : std::log(fx));
                heavyGmmBIC -= 2.0 * ni * (gx == 0.0 ? boost::numeric::bounds<double>::lowest() : std::log(gx));
            }
            LOG_TRACE("light BIC = " << lightGmmBIC << ", heavy BIC = " << heavyGmmBIC);

            if (std::min(lightGmmBIC, heavyGmmBIC) < categoricalBIC) {
                return true;
            }
        } catch (const std::exception& e) { LOG_ERROR("Failed to compute BIC for " << m << " modes: " << e.what()); }
    }

    return false;
}

template<typename INT>
maths::COrdinal CDataSemantics::addInteger(INT value) {
    m_NumericProportion = (m_NumericProportion * (m_Count - 1.0) + 1.0) / m_Count;
    m_IntegerProportion = (m_IntegerProportion * (m_Count - 1.0) + 1.0) / m_Count;
    return maths::COrdinal(static_cast<int64_t>(value));
}

template<typename UINT>
maths::COrdinal CDataSemantics::addPositiveInteger(UINT value) {
    m_NumericProportion = (m_NumericProportion * (m_Count - 1.0) + 1.0) / m_Count;
    m_IntegerProportion = (m_IntegerProportion * (m_Count - 1.0) + 1.0) / m_Count;
    return maths::COrdinal(static_cast<uint64_t>(value));
}

template<typename REAL>
maths::COrdinal CDataSemantics::addReal(REAL value) {
    m_NumericProportion = (m_NumericProportion * (m_Count - 1.0) + 1.0) / m_Count;
    return maths::COrdinal(static_cast<double>(value));
}

const std::size_t CDataSemantics::MAXIMUM_EMPIRICAL_DISTRIBUTION_SIZE(10000);
const double CDataSemantics::NUMERIC_PROPORTION_FOR_METRIC_STRICT(0.99);
const double CDataSemantics::NUMERIC_PROPORTION_FOR_METRIC_WITH_SUSPECTED_MISSING_VALUES(0.5);
const double CDataSemantics::INTEGER_PRORORTION_FOR_INTEGER(0.999);
}
}
