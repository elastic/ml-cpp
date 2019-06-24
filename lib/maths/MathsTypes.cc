/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/MathsTypes.h>

#include <maths/CMathsFuncs.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <boost/math/special_functions/fpclassify.hpp>

#include <stdexcept>

namespace ml {
namespace maths_t {
namespace {
TDoubleWeightsAry unitWeight() {
    TDoubleWeightsAry result;
    result.fill(1.0);
    return result;
}
}

const TDoubleWeightsAry CUnitWeights::UNIT(unitWeight());
const TDoubleWeightsAry1Vec CUnitWeights::SINGLE_UNIT{unitWeight()};

TDoubleWeightsAry countWeight(double weight) {
    TDoubleWeightsAry result(CUnitWeights::UNIT);
    result[E_SampleCountWeight] = weight;
    return result;
}

TDouble10VecWeightsAry countWeight(double weight, std::size_t dimension) {
    TDouble10VecWeightsAry result(CUnitWeights::unit<TDouble10Vec>(dimension));
    result[E_SampleCountWeight] = TDouble10Vec(dimension, weight);
    return result;
}

void setCount(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights) {
    weights[E_SampleCountWeight] = TDouble10Vec(dimension, weight);
}

double countForUpdate(const TDoubleWeightsAry& weights) {
    return weights[E_SampleCountWeight] * weights[E_SampleWinsorisationWeight];
}

TDouble10Vec countForUpdate(const TDouble10VecWeightsAry& weights) {
    TDouble10Vec result(weights[E_SampleCountWeight]);
    for (std::size_t i = 0u; i < weights[E_SampleWinsorisationWeight].size(); ++i) {
        result[i] *= weights[E_SampleWinsorisationWeight][i];
    }
    return result;
}

TDoubleWeightsAry winsorisationWeight(double weight) {
    TDoubleWeightsAry result(CUnitWeights::UNIT);
    result[E_SampleWinsorisationWeight] = weight;
    return result;
}

TDouble10VecWeightsAry winsorisationWeight(double weight, std::size_t dimension) {
    TDouble10VecWeightsAry result(CUnitWeights::unit<TDouble10Vec>(dimension));
    result[E_SampleWinsorisationWeight] = TDouble10Vec(dimension, weight);
    return result;
}

void setWinsorisationWeight(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights) {
    weights[E_SampleWinsorisationWeight] = TDouble10Vec(dimension, weight);
}

bool isWinsorised(const TDoubleWeightsAry& weights) {
    return weights[E_SampleWinsorisationWeight] != 1.0;
}

bool isWinsorised(const TDoubleWeightsAry1Vec& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TDoubleWeightsAry& weight) {
        return isWinsorised(weight);
    });
}

TDoubleWeightsAry seasonalVarianceScaleWeight(double weight) {
    TDoubleWeightsAry result(CUnitWeights::UNIT);
    result[E_SampleSeasonalVarianceScaleWeight] = weight;
    return result;
}

TDouble10VecWeightsAry seasonalVarianceScaleWeight(double weight, std::size_t dimension) {
    TDouble10VecWeightsAry result(CUnitWeights::unit<TDouble10Vec>(dimension));
    result[E_SampleSeasonalVarianceScaleWeight] = TDouble10Vec(dimension, weight);
    return result;
}

void setSeasonalVarianceScale(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights) {
    weights[E_SampleSeasonalVarianceScaleWeight] = TDouble10Vec(dimension, weight);
}

bool hasSeasonalVarianceScale(const TDoubleWeightsAry& weights) {
    return weights[E_SampleSeasonalVarianceScaleWeight] != 1.0;
}

bool hasSeasonalVarianceScale(const TDoubleWeightsAry1Vec& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TDoubleWeightsAry& weight) {
        return hasSeasonalVarianceScale(weight);
    });
}

TDoubleWeightsAry countVarianceScaleWeight(double weight) {
    TDoubleWeightsAry result(CUnitWeights::UNIT);
    result[E_SampleCountVarianceScaleWeight] = weight;
    return result;
}

TDouble10VecWeightsAry countVarianceScaleWeight(double weight, std::size_t dimension) {
    TDouble10VecWeightsAry result(CUnitWeights::unit<TDouble10Vec>(dimension));
    result[E_SampleCountVarianceScaleWeight] = TDouble10Vec(dimension, weight);
    return result;
}

void setCountVarianceScale(double weight, std::size_t dimension, TDouble10VecWeightsAry& weights) {
    weights[E_SampleCountVarianceScaleWeight] = TDouble10Vec(dimension, weight);
}

bool hasCountVarianceScale(const TDoubleWeightsAry& weights) {
    return weights[E_SampleCountVarianceScaleWeight] != 1.0;
}

bool hasCountVarianceScale(const TDoubleWeightsAry1Vec& weights) {
    return std::any_of(weights.begin(), weights.end(), [](const TDoubleWeightsAry& weight) {
        return hasCountVarianceScale(weight);
    });
}
}
}
