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

#include <test/CRandomNumbers.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <test/CRandomNumbersDetail.h>

#include <boost/range.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/lognormal_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/poisson_distribution.hpp>
#include <boost/random/student_t_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <numeric>

namespace ml {
namespace test {

void CRandomNumbers::generateNormalSamples(double mean,
                                           double variance,
                                           std::size_t numberSamples,
                                           TDoubleVec &samples) {
    boost::random::normal_distribution<> normal(mean, ::sqrt(variance));
    generateSamples(m_Generator, normal, numberSamples, samples);
}

void CRandomNumbers::generateMultivariateNormalSamples(const TDoubleVec &mean,
                                                       const TDoubleVecVec &covariances_,
                                                       std::size_t numberSamples,
                                                       TDoubleVecVec &samples) {
    samples.clear();

    std::size_t d = covariances_.size();

    Eigen::MatrixXd covariances(d, d);
    for (std::size_t i = 0u; i < d; ++i) {
        for (std::size_t j = 0u; j < d; ++j) {
            covariances(i, j) = covariances_[i][j];
        }
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariances, Eigen::ComputeThinU | Eigen::ComputeThinV);

    std::size_t r = static_cast<std::size_t>(svd.rank());

    TDoubleVecVec residuals(r);
    for (std::size_t i = 0u; i < r; ++i) {
        this->generateNormalSamples(0.0, svd.singularValues()(i), numberSamples, residuals[i]);
    }

    Eigen::VectorXd ri(d);
    TDoubleVec      xi(d, 0.0);
    for (std::size_t i = 0u; i < numberSamples; ++i) {
        for (std::size_t j = 0u; j < r; ++j) {
            ri(j) = j < r ? residuals[j][i] : 0.0;
        }
        ri = svd.matrixU() * ri;

        for (std::size_t j = 0u; j < r; ++j) {
            xi[j] = mean[j] + ri(j);
        }
        samples.push_back(xi);
    }
}

void CRandomNumbers::generatePoissonSamples(double rate,
                                            std::size_t numberSamples,
                                            TUIntVec &samples) {
    boost::random::poisson_distribution<> poisson(rate);
    generateSamples(m_Generator, poisson, numberSamples, samples);
}

void CRandomNumbers::generateStudentsSamples(double degreesFreedom,
                                             std::size_t numberSamples,
                                             TDoubleVec &samples) {
    boost::random::student_t_distribution<> students(degreesFreedom);
    generateSamples(m_Generator, students, numberSamples, samples);
}

void CRandomNumbers::generateLogNormalSamples(double location,
                                              double squareScale,
                                              std::size_t numberSamples,
                                              TDoubleVec &samples) {
    boost::random::lognormal_distribution<> logNormal(location, ::sqrt(squareScale));
    generateSamples(m_Generator, logNormal, numberSamples, samples);
}

void CRandomNumbers::generateUniformSamples(double a,
                                            double b,
                                            std::size_t numberSamples,
                                            TDoubleVec &samples) {
    boost::random::uniform_real_distribution<> uniform(a, b);
    generateSamples(m_Generator, uniform, numberSamples, samples);
}

void CRandomNumbers::generateUniformSamples(std::size_t a,
                                            std::size_t b,
                                            std::size_t numberSamples,
                                            TSizeVec &samples) {
    boost::random::uniform_int_distribution<std::size_t> uniform(a, b - 1);
    generateSamples(m_Generator, uniform, numberSamples, samples);
}

void CRandomNumbers::generateGammaSamples(double shape,
                                          double scale,
                                          std::size_t numberSamples,
                                          TDoubleVec &samples) {
    boost::random::gamma_distribution<> gamma(shape, scale);
    generateSamples(m_Generator, gamma, numberSamples, samples);
}

void CRandomNumbers::generateMultinomialSamples(const TDoubleVec &categories,
                                                const TDoubleVec &probabilities,
                                                std::size_t numberSamples,
                                                TDoubleVec &samples) {
    if (categories.size() != probabilities.size()) {
        LOG_ERROR("categories and probabilities must be one-to-one.");
    }

    // We use inverse transform sampling to generate the mutinomial
    // samples from a collection of random samples on [0,1].

    boost::random::uniform_real_distribution<> uniform(0.0, 1.0);
    generateSamples(m_Generator, uniform, numberSamples, samples);

    // Construct the transform function.
    TDoubleVec transform;
    transform.reserve(probabilities.size());
    std::partial_sum(probabilities.begin(),
                     probabilities.end(),
                     std::back_inserter(transform));

    // Map the samples to categories.
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        std::size_t j = std::lower_bound(transform.begin(),
                                         transform.end(),
                                         samples[i]) - transform.begin();
        if (j == transform.size()) {
            LOG_ERROR("Expected sample " << samples[i]
                                         << " to be less than largest value in "
                                         << core::CContainerPrinter::print(transform));
            j = transform.size() - 1;
        }
        samples[i] = categories[j];
    }
}

void CRandomNumbers::generateDirichletSamples(const TDoubleVec &concentrations,
                                              std::size_t numberSamples,
                                              TDoubleVecVec &samples) {
    samples.resize(numberSamples);
    for (std::size_t i = 0; i < concentrations.size(); ++i) {
        TDoubleVec raw;
        generateGammaSamples(concentrations[i], 1.0, numberSamples, raw);
        for (std::size_t j = 0u; j < numberSamples; ++j) {
            samples[j].reserve(concentrations.size());
            samples[j].push_back(raw[j]);
        }
    }

    for (std::size_t i = 0u; i < samples.size(); ++i) {
        double normalizer = 0.0;
        for (std::size_t j = 0u; j < concentrations.size(); ++j) {
            normalizer += samples[i][j];
        }
        for (std::size_t j = 0u; j < samples[i].size(); ++j) {
            samples[i][j] /= normalizer;
        }
    }
}

void CRandomNumbers::generateWords(std::size_t length,
                                   std::size_t numberSamples,
                                   TStrVec &samples) {
    const char characterSet[] =
    {
        'a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j',
        'k', 'l', 'm', 'n', 'o',
        'p', 'q', 'r', 's', 't',
        'u', 'v', 'x', 'y', 'z',
        '-', '_', ' ', '1', '2',
        '3', '4', '5', '6', '7',
        '8', '9', '0'
    };

    boost::random::uniform_int_distribution<size_t>
    uniform(0u, boost::size(characterSet) - 1);

    samples.resize(numberSamples);
    for (std::size_t i = 0u; i < numberSamples; ++i) {
        std::string &word = samples[i];
        word.resize(length);
        for (std::size_t j = 0u; j < length; ++j) {
            word[j] = characterSet[uniform(m_Generator)];
        }
    }
}

CRandomNumbers::CUniform0nGenerator CRandomNumbers::uniformGenerator(void) {
    return CUniform0nGenerator(m_Generator);
}

void CRandomNumbers::discard(std::size_t n) {
    m_Generator.discard(n);
}

CRandomNumbers::CUniform0nGenerator::CUniform0nGenerator(const TGenerator &generator) :
    m_Generator(new TGenerator(generator)) {}

std::size_t CRandomNumbers::CUniform0nGenerator::operator()(std::size_t n) const {
    boost::random::uniform_int_distribution<std::size_t> uniform(0, n - 1);
    return uniform(*m_Generator);
}

}
}
