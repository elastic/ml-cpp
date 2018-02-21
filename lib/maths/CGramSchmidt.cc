/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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

bool CGramSchmidt::check(const TDoubleVecVec &x)
{
    if (!x.empty())
    {
        std::size_t dimension{x[0].size()};
        for (std::size_t i = 1u; i < x.size(); ++i)
        {
            if (x[i].size() != dimension)
            {
                return false;
            }
        }
    }
    return true;
}

void CGramSchmidt::swap(TDoubleVec &x, TDoubleVec &y)
{
    x.swap(y);
}

void CGramSchmidt::minusProjection(TDoubleVec &x, const TDoubleVec &e)
{
    double n{inner(x, e)};
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        x[i] -= n * e[i];
    }
}

void CGramSchmidt::divide(TDoubleVec &x, double s)
{
    for (auto &&coordinate : x)
    {
        coordinate /= s;
    }
}

double CGramSchmidt::norm(const TDoubleVec &x)
{
    return ::sqrt(inner(x, x));
}

double CGramSchmidt::inner(const TDoubleVec &x, const TDoubleVec &y)
{
    double result{0.0};
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        result += x[i] * y[i];
    }
    return result;
}

void CGramSchmidt::zero(TDoubleVec &x)
{
    std::fill(x.begin(), x.end(), 0.0);
}

std::string CGramSchmidt::print(const TDoubleVec &x)
{
    return core::CContainerPrinter::print(x);
}

}
}
