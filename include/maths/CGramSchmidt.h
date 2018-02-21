/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CGramSchmidt_h
#define INCLUDED_ml_maths_CGramSchmidt_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CNonInstantiatable.h>

#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/ImportExport.h>

#include <boost/array.hpp>

#include <limits>
#include <ostream>
#include <vector>

namespace ml
{
namespace maths
{

//! \brief Computes an orthonormal basis for a collection
//! of vectors using the Gram-Schmidt process.
//!
//! DESCRIPTION:\n
//! See https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
class MATHS_EXPORT CGramSchmidt : private core::CNonInstantiatable
{
    public:
        using TDoubleVec = std::vector<double>;
        using TDoubleVecVec = std::vector<TDoubleVec>;

    public:
        //! Compute an orthonormal basis for the vectors in \p x.
        //!
        //! \param[in,out] x The vectors from which to compute the
        //! orthonormal basis. Overwritten with the basis.
        static bool basis(TDoubleVecVec &x);

        //! Compute an orthonormal basis for the vectors in \p x.
        //!
        //! \param[in,out] x The vectors from which to compute the
        //! orthonormal basis. Overwritten with the basis.
        template<std::size_t N>
        static bool basis(boost::array<TDoubleVec, N> &x)
        {
            return basisImpl(x);
        }

        //! Compute an orthonormal basis for the vectors in \p x.
        //!
        //! \param[in,out] x The vectors from which to compute the
        //! orthonormal basis. Overwritten with the basis.
        template<typename VECTOR>
        static bool basis(std::vector<VECTOR> &x)
        {
            return basisImpl(x);
        }

        //! Compute an orthonormal basis for the vectors in \p x.
        //!
        //! \param[in,out] x The vectors from which to compute the
        //! orthonormal basis. Overwritten with the basis.
        template<typename VECTOR, std::size_t N>
        static bool basis(boost::array<VECTOR, N> &x)
        {
            return basisImpl(x);
        }

    private:
        //! The Gram-Schmidt process.
        //!
        //! \param[in,out] x The vectors from which to compute the
        //! orthonormal basis. Overwritten with the basis.
        template<typename VECTORS>
        static bool basisImpl(VECTORS &x)
        {
            if (!check(x))
            {
                return false;
            }

            std::size_t i{0u};
            std::size_t current{0u};

            for (/**/; i < x.size(); ++i)
            {
                if (i != current)
                {
                    swap(x[current], x[i]);
                }

                double n{norm(x[current])};
                LOG_TRACE("i = " << i
                          << ", current = " << current
                          << ", x = " << print(x[current])
                          << ", norm = " << n);

                if (n != 0.0)
                {
                    divide(x[current], n);
                    ++current; ++i;
                    break;
                }
            }

            try
            {
                for (/**/; i < x.size(); ++i)
                {
                    if (i != current)
                    {
                        swap(x[current], x[i]);
                    }

                    double eps{  5.0
                               * norm(x[current])
                               * std::numeric_limits<double>::epsilon()};

                    for (std::size_t j = 0u; j < i; ++j)
                    {
                        minusProjection(x[current], x[j]);
                    }

                    double n{norm(x[current])};
                    LOG_TRACE("i = " << i
                              << ", current = " << current
                              << ", x = " << print(x[current])
                              << ", norm = " << n
                              << ", eps = " << eps);

                    if (::fabs(n) > eps)
                    {
                        divide(x[current], n);
                        ++current;
                    }
                }

                if (current != x.size())
                {
                    erase(x, x.begin() + current, x.end());
                }
            }
            catch (const std::runtime_error &e)
            {
                LOG_ERROR("Failed to construct basis: " << e.what());
                return false;
            }
            return true;
        }

        //! Check all the vectors have the same dimension.
        static bool check(const TDoubleVecVec &x);

        //! Check all the vectors have the same dimension.
        template<typename VECTORS>
        static bool check(const VECTORS &x)
        {
            if (!x.empty())
            {
                std::size_t dimension{las::dimension(x[0])};
                for (std::size_t i = 1u; i < x.size(); ++i)
                {
                    if (las::dimension(x[i]) != dimension)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        //! Efficiently swap \p x and \p y.
        static void swap(TDoubleVec &x, TDoubleVec &y);

        //! Efficiently swap \p x and \p y.
        template<typename VECTOR>
        static void swap(VECTOR &x, VECTOR &y)
        {
            using std::swap;
            swap(x, y);
        }

        //! Subtract the projection of \p x onto \p e from \p x.
        static void minusProjection(TDoubleVec &x, const TDoubleVec &e);

        //! Subtract the projection of \p x onto \p e from \p x.
        template<typename VECTOR>
        static void minusProjection(VECTOR &x, const VECTOR &e)
        {
            typename SCoordinate<VECTOR>::Type n{las::inner(e, x)};
            x -= n * e;
        }

        //! Divide the vector \p x by the scalar \p s.
        static void divide(TDoubleVec &x, double s);

        //! Divide the vector \p x by the scalar \p s.
        template<typename VECTOR>
        static void divide(VECTOR &x, double s)
        {
            x /= s;
        }

        //! Compute the norm of the vector \p x.
        static double norm(const TDoubleVec &x);

        //! Compute the norm of the vector \p x.
        template<typename VECTOR>
        static double norm(const VECTOR &x)
        {
            return las::norm(x);
        }

        //! Compute the inner product of \p x and \p y.
        static double inner(const TDoubleVec &x, const TDoubleVec &y);

        //! Compute the inner product of \p x and \p y.
        template<typename VECTOR>
        static double inner(const VECTOR &x, const VECTOR &y)
        {
            return las::inner(x, y);
        }

        //! Remove [\p begin, \p end) from \p x.
        template<typename VECTOR>
        static void erase(std::vector<VECTOR> &x,
                          typename std::vector<VECTOR>::iterator begin,
                          typename std::vector<VECTOR>::iterator end)
        {
            x.erase(begin, end);
        }

        //! Remove [\p begin, \p end) from \p x.
        template<typename VECTOR, std::size_t N>
        static void erase(boost::array<VECTOR, N> &/*x*/,
                          typename boost::array<VECTOR, N>::iterator begin,
                          typename boost::array<VECTOR, N>::iterator end)
        {
            for (/**/; begin != end; ++begin)
            {
                zero(*begin);
            }
        }

        //! Zero the components of \p x.
        static void zero(TDoubleVec &x);

        //! Zero the components of \p x.
        template<typename VECTOR>
        static void zero(VECTOR &x)
        {
            x = las::zero(x);
        }

        //! Print \p x for debug.
        static std::string print(const TDoubleVec &x);

        //! Print \p x for debug.
        template<typename VECTOR>
        static std::string print(const VECTOR &x)
        {
            std::ostringstream result;
            result << x;
            return result.str();
        }
};

}
}

#endif // INCLUDED_ml_maths_CGramSchmidt_h
