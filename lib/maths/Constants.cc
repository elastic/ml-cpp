/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <maths/Constants.h>

namespace ml {
namespace maths {

const maths_t::TWeightStyleVec CConstantWeights::COUNT{maths_t::E_SampleCountWeight};
const maths_t::TWeightStyleVec CConstantWeights::COUNT_VARIANCE{
    maths_t::E_SampleCountVarianceScaleWeight};
const maths_t::TWeightStyleVec CConstantWeights::SEASONAL_VARIANCE{
    maths_t::E_SampleSeasonalVarianceScaleWeight};
const CConstantWeights::TDouble4Vec CConstantWeights::UNIT{1.0};
const CConstantWeights::TDouble4Vec1Vec CConstantWeights::SINGLE_UNIT{UNIT};

double maxModelPenalty(double numberSamples) {
    return 10.0 + numberSamples;
}
}
}
