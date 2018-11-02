/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_COrthogonaliser_h
#define INCLUDED_ml_maths_COrthogonaliser_h

#include <core/CLogger.h>

#include <maths/CLinearAlgebraEigen.h>
#include <maths/CLinearAlgebraShims.h>

#include <boost/array.hpp>

#include <vector>

namespace ml {
namespace maths {

//! \brief Orthogonalizes a set of vectors via Househlder Orthogonalisation.
class COrthogonaliser {
public:
    //! Get an orthonormal basis spanning \p vectors. Note that the result is written
    //! back to \p vectors.
    //!
    //! \param[in,out] vectors The vectors for which to compute an orthonormal basis.
    //! Note if there linear dependenices this will resized to the size of the spanning
    //! basis.
    //! \return True if there is an orthonormal basis.
    template<typename VECTOR>
    static bool orthonormalBasis(std::vector<VECTOR>& vectors) {
        return orthonormalBasisImpl(vectors);
    }

    //! Get an orthonormal basis spanning \p vectors. Note that the result is written
    //! back to \p vectors.
    //!
    //! \param[in,out] vectors The linearly independent vectors for which to compute
    //! an orthonormal basis.
    //! \return True if and only if there is an orthonormal basis of size N.
    template<typename VECTOR, std::size_t N>
    static bool orthonormalBasis(boost::array<VECTOR, N>& vectors) {
        return orthonormalBasisImpl(vectors);
    }

private:
    template<typename VECTORS>
    static bool orthonormalBasisImpl(VECTORS& vectors) {
        if (vectors.empty()) {
            return true;
        }

        int dimension{COrthogonaliser::dimension(vectors)};
        if (dimension < 0) {
            return false;
        }

        int rows{dimension};
        int cols{static_cast<int>(vectors.size())};
        CDenseMatrix<double> x(rows, cols);
        for (std::size_t j = 0; j < vectors.size(); ++j) {
            writeColumn(j, vectors[j], x);
        }

        auto qr = x.colPivHouseholderQr();
        long int rank{qr.rank()};
        CDenseMatrix<double> thinQ{CDenseMatrix<double>::Identity(rows, rank)};
        thinQ = qr.householderQ() * thinQ;

        if (resize(vectors, rank) == false) {
            return false;
        }

        for (long int j = 0; j < rank; ++j) {
            readColumn(j, thinQ, vectors[j]);
        }

        return true;
    }

    template<typename VECTORS>
    static int dimension(const VECTORS& vectors) {
        std::size_t result{las::dimension(vectors[0])};
        for (std::size_t i = 1; i < vectors.size(); ++i) {
            if (las::dimension(vectors[i]) != result) {
                LOG_ERROR(<< "Mismatched dimensions " << las::dimension(vectors[i]) << " != " << result);
                return -1;
            }
        }
        return static_cast<int>(result);
    }
    template<typename SCALAR>
    static int dimension(const std::vector<std::vector<SCALAR>>& vectors) {
        std::size_t result{vectors[0].size()};
        for (std::size_t i = 1; i < vectors.size(); ++i) {
            if (vectors[i].size() != result) {
                LOG_ERROR(<< "Mismatched dimensions " << vectors[i].size() << " != " << result);
                return -1;
            }
        }
        return static_cast<int>(result);
    }

    template<typename VECTOR>
    static bool resize(std::vector<VECTOR>& vectors, std::size_t rank) {
        vectors.resize(rank);
        return true;
    }
    template<typename VECTOR, std::size_t N>
    static bool resize(boost::array<VECTOR, N>&, std::size_t) {
        LOG_ERROR(<< "Expecting no linear dependencies");
        return false;
    }

    template<typename VECTOR>
    static void writeColumn(std::size_t j, const VECTOR& vector, CDenseMatrix<double>& m) {
        for (std::size_t i = 0; i < las::dimension(vector); ++i) {
            m(i, j) = vector(i);
        }
    }
    template<typename SCALAR>
    static void writeColumn(std::size_t j, const std::vector<SCALAR>& vector, CDenseMatrix<double>& m) {
        for (std::size_t i = 0; i < vector.size(); ++i) {
            m(i, j) = vector[i];
        }
    }
    template<typename SCALAR>
    static void writeColumn(std::size_t j, const CDenseVector<SCALAR>& vector, CDenseMatrix<double>& m) {
        m.col(j) = vector;
    }

    template<typename VECTOR>
    static void readColumn(std::size_t j, const CDenseMatrix<double>& m, VECTOR& vector) {
        for (std::size_t i = 0; i < las::dimension(vector); ++i) {
            vector(i) = m(i, j);
        }
    }
    template<typename SCALAR>
    static void readColumn(std::size_t j, const CDenseMatrix<double>& m, CDenseVector<SCALAR>& vector) {
        vector = m.col(j);
    }
    template<typename SCALAR>
    static void readColumn(std::size_t j, const CDenseMatrix<double>& m, std::vector<SCALAR>& vector) {
        for (std::size_t i = 0; i < vector.size(); ++i) {
            vector[i] = m(i, j);
        }
    }
};
}
}

#endif // INCLUDED_ml_maths_COrthogonaliser_h
