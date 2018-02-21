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

#include <maths/MathsTypes.h>

#include <maths/CMathsFuncs.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stdexcept>

namespace ml
{
namespace maths_t
{
namespace
{
namespace detail
{

//! Check that the weights styles and weights are consistent.
template<typename T>
inline bool check(const TWeightStyleVec &weightStyles,
                  const core::CSmallVector<T, 4> &weights)
{
    if (weightStyles.size() == weights.size())
    {
        return true;
    }
    LOG_ERROR("Mismatch in weight styles '"
              << core::CContainerPrinter::print(weightStyles)
              << "' and weights '"
              << core::CContainerPrinter::print(weights) << "'");
    return false;
}

//! Multiply \p lhs by \p rhs.
inline void multiplyEquals(double rhs, double &lhs)
{
    lhs *= rhs;
}

//! Elementwise multiply \p lhs by \p rhs.
inline void multiplyEquals(const TDouble10Vec &rhs, TDouble10Vec &lhs)
{
    for (std::size_t i = 0u; i < lhs.size(); ++i)
    {
        lhs[i] *= rhs[i];
    }
}

//! Check if less than zero.
inline bool isNegative(double value)
{
    return value < 0.0;
}

//! Elementwise check if less than zero.
inline bool isNegative(const TDouble10Vec &values)
{
    for (auto value : values)
    {
        if (value < 0.0)
        {
            return true;
        }
    }
    return false;
}

//! Check if less than or equal to zero.
inline bool isNonPostive(double value)
{
    return value <= 0.0;
}

//! Elementwise check if less than or equal to zero.
inline bool isNonPostive(const TDouble10Vec &values)
{
    for (auto value : values)
    {
        if (value < 0.0)
        {
            return true;
        }
    }
    return false;
}

//! Extract the effective sample count from a collection of weights.
template<typename T>
void count(const TWeightStyleVec &weightStyles,
           const core::CSmallVector<T, 4> &weights,
           T &result)
{
    if (check(weightStyles, weights))
    {
        T candidate(result);
        for (std::size_t i = 0u; i < weightStyles.size(); ++i)
        {
            switch (weightStyles[i])
            {
            case E_SampleCountWeight:                 multiplyEquals(weights[i], candidate); break;
            case E_SampleSeasonalVarianceScaleWeight: break;
            case E_SampleCountVarianceScaleWeight:    break;
            case E_SampleWinsorisationWeight:         break;
            }
        }
        if (!maths::CMathsFuncs::isFinite(result) || isNegative(result))
        {
            LOG_ERROR("Ignoring bad count weight: " << result);
        }
        else
        {
            result = std::move(candidate);
        }
    }
}

//! Extract the effective sample count with which to update a model
//! from a collection of weights.
template<typename T>
void countForUpdate(const TWeightStyleVec &weightStyles,
                    const core::CSmallVector<T, 4> &weights,
                    T &result)
{
    if (check(weightStyles, weights))
    {
        T candidate(result);
        for (std::size_t i = 0u; i < weightStyles.size(); ++i)
        {
            switch (weightStyles[i])
            {
            case E_SampleCountWeight:                 multiplyEquals(weights[i], candidate); break;
            case E_SampleSeasonalVarianceScaleWeight: break;
            case E_SampleCountVarianceScaleWeight:    break;
            case E_SampleWinsorisationWeight:         multiplyEquals(weights[i], candidate); break;
            }
        }
        if (!maths::CMathsFuncs::isFinite(result) || isNegative(result))
        {
            LOG_ERROR("Ignoring bad count weight: " << result);
        }
        else
        {
            result = std::move(candidate);
        }
    }
}

//! Extract the Winsorisation weight from a collection of weights.
template<typename T>
void winsorisationWeight(const TWeightStyleVec &weightStyles,
                         const core::CSmallVector<T, 4> &weights,
                         T &result)
{
    if (check(weightStyles, weights))
    {
        T candidate(result);
        for (std::size_t i = 0u; i < weightStyles.size(); ++i)
        {
            switch (weightStyles[i])
            {
            case E_SampleCountWeight:                 break;
            case E_SampleSeasonalVarianceScaleWeight: break;
            case E_SampleCountVarianceScaleWeight:    break;
            case E_SampleWinsorisationWeight:         multiplyEquals(weights[i], candidate); break;
            }
        }
        if (!maths::CMathsFuncs::isFinite(result) || isNegative(result))
        {
            LOG_ERROR("Ignoring bad Winsorisation weight: " << result);
        }
        else
        {
            result = std::move(candidate);
        }
    }
}

//! Extract the seasonal variance scale from a collection of weights.
template<typename T>
void seasonalVarianceScale(const TWeightStyleVec &weightStyles,
                           const core::CSmallVector<T, 4> &weights,
                           T &result)
{
    if (check(weightStyles, weights))
    {
        T candidate(result);
        for (std::size_t i = 0u; i < weightStyles.size(); ++i)
        {
            switch (weightStyles[i])
            {
            case E_SampleCountWeight:                 break;
            case E_SampleSeasonalVarianceScaleWeight: multiplyEquals(weights[i], candidate); break;
            case E_SampleCountVarianceScaleWeight:    break;
            case E_SampleWinsorisationWeight:         break;
            }
        }
        if (!maths::CMathsFuncs::isFinite(result) || isNonPostive(result))
        {
            LOG_ERROR("Ignoring bad variance scale: " << result);
        }
        else
        {
            result = std::move(candidate);
        }
    }
}

//! Extract the count variance scale from a collection of weights.
template<typename T>
void countVarianceScale(const TWeightStyleVec &weightStyles,
                        const core::CSmallVector<T, 4> &weights,
                        T &result)
{
    if (check(weightStyles, weights))
    {
        T candidate(result);
        for (std::size_t i = 0u; i < weightStyles.size(); ++i)
        {
            switch (weightStyles[i])
            {
            case E_SampleCountWeight:                 break;
            case E_SampleSeasonalVarianceScaleWeight: break;
            case E_SampleCountVarianceScaleWeight:    multiplyEquals(weights[i], candidate); break;
            case E_SampleWinsorisationWeight:         break;
            }
        }
        if (!maths::CMathsFuncs::isFinite(result) || isNonPostive(result))
        {
            LOG_ERROR("Ignoring bad variance scale: " << result);
        }
        else
        {
            result = std::move(candidate);
        }
    }
}

}
}

double count(const TWeightStyleVec &weightStyles,
             const TDouble4Vec &weights)
{
    double result{1.0};
    detail::count(weightStyles, weights, result);
    return result;
}

TDouble10Vec count(std::size_t dimension,
                   const TWeightStyleVec &weightStyles,
                   const TDouble10Vec4Vec &weights)
{
    TDouble10Vec result(dimension, 1.0);
    detail::count(weightStyles, weights, result);
    return result;

}

double countForUpdate(const TWeightStyleVec &weightStyles,
                      const TDouble4Vec &weights)
{
    double result{1.0};
    detail::countForUpdate(weightStyles, weights, result);
    return result;
}

TDouble10Vec countForUpdate(std::size_t dimension,
                            const TWeightStyleVec &weightStyles,
                            const TDouble10Vec4Vec &weights)
{
    TDouble10Vec result(dimension, 1.0);
    detail::countForUpdate(weightStyles, weights, result);
    return result;
}

double winsorisationWeight(const TWeightStyleVec &weightStyles,
                           const TDouble4Vec &weights)
{
    double result{1.0};
    detail::winsorisationWeight(weightStyles, weights, result);
    return result;
}

TDouble10Vec winsorisationWeight(std::size_t dimension,
                                 const TWeightStyleVec &weightStyles,
                                 const TDouble10Vec4Vec &weights)
{
    TDouble10Vec result(dimension, 1.0);
    detail::winsorisationWeight(weightStyles, weights, result);
    return result;
}

double seasonalVarianceScale(const TWeightStyleVec &weightStyles,
                             const TDouble4Vec &weights)
{
    double result{1.0};
    detail::seasonalVarianceScale(weightStyles, weights, result);
    return result;
}

TDouble10Vec seasonalVarianceScale(std::size_t dimension,
                                   const TWeightStyleVec &weightStyles,
                                   const TDouble10Vec4Vec &weights)
{
    TDouble10Vec result(dimension, 1.0);
    detail::seasonalVarianceScale(weightStyles, weights, result);
    return result;
}

double countVarianceScale(const TWeightStyleVec &weightStyles,
                          const TDouble4Vec &weights)
{
    double result{1.0};
    detail::countVarianceScale(weightStyles, weights, result);
    return result;
}

TDouble10Vec countVarianceScale(std::size_t dimension,
                                const TWeightStyleVec &weightStyles,
                                const TDouble10Vec4Vec &weights)
{
    TDouble10Vec result(dimension, 1.0);
    detail::countVarianceScale(weightStyles, weights, result);
    return result;
}

bool hasSeasonalVarianceScale(const TWeightStyleVec &weightStyles,
                              const TDouble4Vec &weights)
{
    return seasonalVarianceScale(weightStyles, weights) != 1.0;
}

bool hasSeasonalVarianceScale(const TWeightStyleVec &weightStyles,
                              const TDouble4Vec1Vec &weights)
{
    for (std::size_t i = 0u; i < weights.size(); ++i)
    {
        if (hasSeasonalVarianceScale(weightStyles, weights[i]))
        {
            return true;
        }
    }
    return false;
}

bool hasSeasonalVarianceScale(const TWeightStyleVec &weightStyles,
                              const TDouble10Vec4Vec &weights)
{
    if (!detail::check(weightStyles, weights))
    {
        return false;
    }
    for (std::size_t i = 0u; i < weightStyles.size(); ++i)
    {
        switch (weightStyles[i])
        {
        case E_SampleCountWeight:
            break;
        case E_SampleSeasonalVarianceScaleWeight:
            for (std::size_t j = 0u; j < weights[i].size(); ++j)
            {
                if (weights[i][j] != 1.0)
                {
                    return true;
                }
            }
            break;
        case E_SampleCountVarianceScaleWeight:
            break;
        case E_SampleWinsorisationWeight:
            break;
        }
    }
    return false;
}

bool hasSeasonalVarianceScale(const TWeightStyleVec &weightStyles,
                              const TDouble10Vec4Vec1Vec &weights)
{
    for (std::size_t i = 0u; i < weights.size(); ++i)
    {
        if (hasSeasonalVarianceScale(weightStyles, weights[i]))
        {
            return true;
        }
    }
    return false;
}

bool hasCountVarianceScale(const TWeightStyleVec &weightStyles,
                           const TDouble4Vec &weights)
{
    return countVarianceScale(weightStyles, weights) != 1.0;
}

bool hasCountVarianceScale(const TWeightStyleVec &weightStyles,
                           const TDouble4Vec1Vec &weights)
{
    for (std::size_t i = 0u; i < weights.size(); ++i)
    {
        if (hasCountVarianceScale(weightStyles, weights[i]))
        {
            return true;
        }
    }
    return false;
}

bool hasCountVarianceScale(const TWeightStyleVec &weightStyles,
                           const TDouble10Vec4Vec &weights)
{
    if (!detail::check(weightStyles, weights))
    {
        return false;
    }
    for (std::size_t i = 0u; i < weightStyles.size(); ++i)
    {
        switch (weightStyles[i])
        {
        case E_SampleCountWeight:
            break;
        case E_SampleSeasonalVarianceScaleWeight:
            break;
        case E_SampleCountVarianceScaleWeight:
            for (std::size_t j = 0u; j < weights[i].size(); ++j)
            {
                if (weights[i][j] != 1.0)
                {
                    return true;
                }
            }
            break;
        case E_SampleWinsorisationWeight:
            break;
        }
    }
    return false;
}

bool hasCountVarianceScale(const TWeightStyleVec &weightStyles,
                           const TDouble10Vec4Vec1Vec &weights)
{
    for (std::size_t i = 0u; i < weights.size(); ++i)
    {
        if (hasCountVarianceScale(weightStyles, weights[i]))
        {
            return true;
        }
    }
    return false;
}

}
}



