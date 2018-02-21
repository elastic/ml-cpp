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

#include <maths/CGramSchmidt.h>

#include <maths/CLinearAlgebraTools.h>

namespace ml
{
namespace maths
{

bool CGramSchmidt::basis(TDoubleVecVec &x)
{
    return basisImpl(x);
}

bool CGramSchmidt::basis(TVectorVec &x)
{
    return basisImpl(x);
}

void CGramSchmidt::swap(TDoubleVec &x, TDoubleVec &y)
{
    x.swap(y);
}

void CGramSchmidt::swap(TVector &x, TVector &y)
{
    x.swap(y);
}

const CGramSchmidt::TDoubleVec &CGramSchmidt::minusProjection(TDoubleVec &x,
                                                              const TDoubleVec &e)
{
    sameDimension(x, e);
    double n = inner(x, e);
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        x[i] -= n * e[i];
    }
    return x;
}

const CGramSchmidt::TVector &CGramSchmidt::minusProjection(TVector &x,
                                                           const TVector &e)
{
    double n = e.inner(x);
    return x -= n * e;
}

const CGramSchmidt::TDoubleVec &CGramSchmidt::divide(TDoubleVec &x, double s)
{
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        x[i] /= s;
    }
    return x;
}

const CGramSchmidt::TVector &CGramSchmidt::divide(TVector &x, double s)
{
    return x /= s;
}

double CGramSchmidt::norm(const TDoubleVec &x)
{
    return ::sqrt(inner(x, x));
}

double CGramSchmidt::norm(const TVector &x)
{
    return x.euclidean();
}

double CGramSchmidt::inner(const TDoubleVec &x, const TDoubleVec &y)
{
    sameDimension(x, y);
    double result = 0.0;
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        result += x[i] * y[i];
    }
    return result;
}

double CGramSchmidt::inner(const TVector &x, const TVector &y)
{
    sameDimension(x, y);
    return x.inner(y);
}

void CGramSchmidt::sameDimension(const TDoubleVec &x,
                                 const TDoubleVec &y)
{
    if (x.size() != y.size())
    {
        throw std::runtime_error("Mismatching dimensions: "
                                 + core::CStringUtils::typeToString(x.size())
                                 + " != "
                                 + core::CStringUtils::typeToString(y.size()));
    }
}

void CGramSchmidt::sameDimension(const TVector &x,
                                 const TVector &y)
{
    if (x.dimension() != y.dimension())
    {
        throw std::runtime_error("Mismatching dimensions: "
                                 + core::CStringUtils::typeToString(x.dimension())
                                 + " != "
                                 + core::CStringUtils::typeToString(y.dimension()));
    }
}

void CGramSchmidt::zero(TDoubleVec &x)
{
    std::fill(x.begin(), x.end(), 0.0);
}

void CGramSchmidt::zero(TVector &x)
{
    for (std::size_t i = 0u; i < x.dimension(); ++i)
    {
        x(i) = 0.0;
    }
}

std::string CGramSchmidt::print(const TDoubleVec &x)
{
    return core::CContainerPrinter::print(x);
}

std::string CGramSchmidt::print(const TVector &x)
{
    std::ostringstream result;
    result << x;
    return result.str();
}

}
}
