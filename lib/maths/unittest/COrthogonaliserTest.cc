/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/COrthogonaliser.h>

#include <test/CRandomNumbers.h>
#include <test/BoostTestCloseAbsolute.h>

#include <boost/test/unit_test.hpp>
#include <boost/range.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(COrthogonaliserTest)

using namespace ml;

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TVector4 = maths::CVectorNx1<double, 4>;
using TVector4Vec = std::vector<TVector4>;

namespace {

template<typename T>
void generate(test::CRandomNumbers& rng, std::size_t n, std::size_t d, std::vector<T>& x) {
    LOG_DEBUG(<< "n = " << n << ", d = " << d);

    TDoubleVec components;
    rng.generateUniformSamples(0.0, 10.0, n * d, components);

    x.clear();
    for (std::size_t i = 0u; i < n; ++i) {
        x.push_back(T(&components[i * d], &components[(i + 1) * d]));
    }
}

void debug(const TDoubleVecVec& x) {
    LOG_DEBUG(<< "x =");
    for (std::size_t i = 0u; i < x.size(); ++i) {
        LOG_DEBUG(<< "  " << core::CContainerPrinter::print(x[i]));
    }
}

void debug(const TVector4Vec& x) {
    LOG_DEBUG(<< "x =");
    for (std::size_t i = 0u; i < x.size(); ++i) {
        LOG_DEBUG(<< "  " << x[i]);
    }
}

double inner(const TDoubleVec& x, const TDoubleVec& y) {
    BOOST_CHECK_EQUAL(x.size(), y.size());
    double result = 0.0;
    for (std::size_t i = 0u; i < x.size(); ++i) {
        result += x[i] * y[i];
    }
    return result;
}

TDoubleVec multiply(const TDoubleVec& x, double s) {
    TDoubleVec result = x;
    for (std::size_t i = 0u; i < x.size(); ++i) {
        result[i] *= s;
    }
    return result;
}

const TDoubleVec& add(TDoubleVec& x, const TDoubleVec& y) {
    BOOST_CHECK_EQUAL(x.size(), y.size());
    for (std::size_t i = 0u; i < x.size(); ++i) {
        x[i] += y[i];
    }
    return x;
}

const TDoubleVec& subtract(TDoubleVec& x, const TDoubleVec& y) {
    BOOST_CHECK_EQUAL(x.size(), y.size());
    for (std::size_t i = 0u; i < x.size(); ++i) {
        x[i] -= y[i];
    }
    return x;
}
}

BOOST_AUTO_TEST_CASE(testOrthogonality) {
    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "*** Test vector ***");

        TDoubleVecVec x;
        for (std::size_t t = 0u; t < 50; ++t) {
            std::size_t d = t / 5 + 5;
            std::size_t n = t / 5 + 2;

            LOG_DEBUG(<< "n = " << n << ", d = " << d);

            generate(rng, n, d, x);
            maths::COrthogonaliser::orthonormalBasis(x);

            if (t % 10 == 0)
                debug(x);
            for (std::size_t i = 0u; i < x.size(); ++i) {
                for (std::size_t j = i + 1; j < x.size(); ++j) {
                    double xiDotxj = inner(x[i], x[j]);
                    if (t % 10 == 0) {
                        LOG_DEBUG(<< "x(i)' x(j) = " << xiDotxj);
                    }
                    BOOST_CHECK_CLOSE_ABSOLUTE(0.0, xiDotxj, 1e-10);
                }
            }
        }
    }

    {
        LOG_DEBUG(<< "*** Test CVectorNx1 ***");

        TVector4Vec x;
        for (std::size_t t = 0u; t < 50; ++t) {
            generate(rng, 4, 4, x);
            maths::COrthogonaliser::orthonormalBasis(x);

            if (t % 10 == 0)
                debug(x);
            for (std::size_t i = 0u; i < x.size(); ++i) {
                for (std::size_t j = i + 1; j < x.size(); ++j) {
                    double xiDotxj = x[i].inner(x[j]);
                    if (t % 10 == 0) {
                        LOG_DEBUG(<< "x(i)' x(j) = " << xiDotxj);
                    }
                    BOOST_CHECK_CLOSE_ABSOLUTE(0.0, xiDotxj, 1e-10);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testNormalisation) {
    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "*** Test vector ***");

        TDoubleVecVec x;
        for (std::size_t t = 0u; t < 50; ++t) {
            std::size_t d = t / 5 + 5;
            std::size_t n = t / 5 + 2;

            LOG_DEBUG(<< "n = " << n << ", d = " << d);

            generate(rng, n, d, x);
            maths::COrthogonaliser::orthonormalBasis(x);

            if (t % 10 == 0)
                debug(x);
            for (std::size_t i = 0u; i < x.size(); ++i) {
                double normxi = std::sqrt(inner(x[i], x[i]));
                if (t % 10 == 0) {
                    LOG_DEBUG(<< "|| x(i) || = " << normxi);
                }
                BOOST_CHECK_CLOSE_ABSOLUTE(1.0, normxi, 1e-15);
            }
        }
    }

    {
        LOG_DEBUG(<< "*** Test CVectorNx1 ***");

        TVector4Vec x;
        for (std::size_t t = 0u; t < 50; ++t) {
            generate(rng, 4, 4, x);
            maths::COrthogonaliser::orthonormalBasis(x);

            if (t % 10 == 0)
                debug(x);
            for (std::size_t i = 0u; i < x.size(); ++i) {
                for (std::size_t j = i + 1; j < x.size(); ++j) {
                    double normxi = x[i].euclidean();
                    if (t % 10 == 0) {
                        LOG_DEBUG(<< "|| x(i) || = " << normxi);
                    }
                    BOOST_CHECK_CLOSE_ABSOLUTE(1.0, normxi, 1e-15);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testSpan) {
    test::CRandomNumbers rng;

    {
        LOG_DEBUG(<< "*** Test vector ***");

        TDoubleVecVec x;
        TDoubleVecVec basis;
        for (std::size_t t = 0u; t < 50; ++t) {
            std::size_t d = t / 5 + 5;
            std::size_t n = t / 5 + 2;

            LOG_DEBUG(<< "n = " << n << ", d = " << d);

            generate(rng, n, d, x);
            basis = x;
            maths::COrthogonaliser::orthonormalBasis(basis);

            if (t % 10 == 0)
                debug(basis);
            for (std::size_t i = 0u; i < x.size(); ++i) {
                TDoubleVec r(x[i].size(), 0.0);
                for (std::size_t j = 0u; j < basis.size(); ++j) {
                    add(r, multiply(basis[j], inner(x[i], basis[j])));
                }

                if (t % 10 == 0) {
                    LOG_DEBUG(<< "x(i)       = " << core::CContainerPrinter::print(x[i]));
                    LOG_DEBUG(<< "projection = " << core::CContainerPrinter::print(r));
                }

                subtract(r, x[i]);
                double normr = std::sqrt(inner(r, r));

                if (t % 10 == 0) {
                    LOG_DEBUG(<< "|| r || = " << normr);
                }

                BOOST_CHECK_CLOSE_ABSOLUTE(0.0, normr, 1e-12);
            }
        }
    }

    {
        LOG_DEBUG(<< "*** Test CVectorNx1 ***");

        TVector4Vec x;
        TVector4Vec basis;
        for (std::size_t t = 0u; t < 50; ++t) {
            generate(rng, 4, 4, x);
            basis = x;
            maths::COrthogonaliser::orthonormalBasis(basis);

            if (t % 10 == 0)
                debug(x);
            for (std::size_t i = 0u; i < x.size(); ++i) {
                TVector4 r(0.0);
                for (std::size_t j = 0u; j < basis.size(); ++j) {
                    r += basis[j] * x[i].inner(basis[j]);
                }

                if (t % 10 == 0) {
                    LOG_DEBUG(<< "x(i)       = " << x[i]);
                    LOG_DEBUG(<< "projection = " << r);
                }

                r -= x[i];
                double normr = r.euclidean();

                if (t % 10 == 0) {
                    LOG_DEBUG(<< "|| r || = " << normr);
                }

                BOOST_CHECK_CLOSE_ABSOLUTE(0.0, normr, 1e-12);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testEdgeCases) {
    {
        LOG_DEBUG(<< "*** Test zero vector ***");

        TDoubleVecVec x_{{0.0, 0.0, 0.0, 0.0}, {1.0, 3.0, 4.0, 0.0}, {0.4, 0.3, 0.6, 1.0}};
        std::size_t p[]{0, 1, 2};

        do {
            LOG_DEBUG(<< "permutation = " << core::CContainerPrinter::print(p));

            TDoubleVecVec x{x_};
            //debug(x);
            maths::COrthogonaliser::orthonormalBasis(x);
            //debug(x);

            BOOST_CHECK_EQUAL(std::size_t(2), x.size());
        } while (std::next_permutation(p, p + boost::size(p)));
    }
    {
        LOG_DEBUG(<< "");
        LOG_DEBUG(<< "*** Test degenerate ***");

        TDoubleVecVec x_{{1.0, 1.0, 1.0, 1.0},
                         {-1.0, 2.3, 1.0, 0.03},
                         {1.0, 1.0, 1.0, 1.0},
                         {-1.0, 2.3, 1.0, 0.03},
                         {-4.0, 0.3, 1.4, 1.03}};
        std::size_t p[]{0, 1, 2, 3, 4};

        do {
            LOG_DEBUG(<< "permutation = " << core::CContainerPrinter::print(p));

            TDoubleVecVec x{x_};

            //debug(x);
            maths::COrthogonaliser::orthonormalBasis(x);
            //debug(x);

            BOOST_CHECK_EQUAL(std::size_t(3), x.size());
        } while (std::next_permutation(p, p + boost::size(p)));
    }
}


BOOST_AUTO_TEST_SUITE_END()
