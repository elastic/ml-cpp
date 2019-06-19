/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <maths/CPriorStateSerialiser.h>

#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CConstantPrior.h>
#include <maths/CGammaRateConjugate.h>
#include <maths/CLogNormalMeanPrecConjugate.h>
#include <maths/CMultimodalPrior.h>
#include <maths/CMultinomialConjugate.h>
#include <maths/CMultivariateConstantPrior.h>
#include <maths/CMultivariateMultimodalPriorFactory.h>
#include <maths/CMultivariateNormalConjugateFactory.h>
#include <maths/CMultivariateOneOfNPriorFactory.h>
#include <maths/CMultivariatePrior.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/COneOfNPrior.h>
#include <maths/CPoissonMeanConjugate.h>
#include <maths/CPrior.h>


#include <memory>
#include <string>
#include <typeinfo>

namespace ml {
namespace maths {
namespace {
// We use short field names to reduce the state size
// There needs to be one constant here per sub-class of CPrior.
// DO NOT change the existing tags if new sub-classes are added.
const std::string GAMMA_TAG("a");
const std::string LOG_NORMAL_TAG("b");
const std::string MULTIMODAL_TAG("c");
const std::string NORMAL_TAG("d");
const std::string ONE_OF_N_TAG("e");
const std::string POISSON_TAG("f");
const std::string MULTINOMIAL_TAG("g");
const std::string CONSTANT_TAG("h");
const std::string EMPTY_STRING;

//! Implements restore for std::shared_ptr.
template<typename T>
void doRestore(std::shared_ptr<CPrior>& ptr, core::CStateRestoreTraverser& traverser) {
    ptr = std::make_shared<T>(traverser);
}

//! Implements restore for std::unique_ptr.
template<typename T>
void doRestore(std::unique_ptr<CPrior>& ptr, core::CStateRestoreTraverser& traverser) {
    ptr = std::make_unique<T>(traverser);
}

//! Implements restore for std::shared_ptr.
template<typename T>
void doRestore(const SDistributionRestoreParams& params,
               std::shared_ptr<CPrior>& ptr,
               core::CStateRestoreTraverser& traverser) {
    ptr = std::make_shared<T>(params, traverser);
}

//! Implements restore for std::unique_ptr.
template<typename T>
void doRestore(const SDistributionRestoreParams& params,
               std::unique_ptr<CPrior>& ptr,
               core::CStateRestoreTraverser& traverser) {
    ptr = std::make_unique<T>(params, traverser);
}

//! Implements restore into the supplied pointer.
template<typename PTR>
bool restore(const SDistributionRestoreParams& params,
             PTR& ptr,
             core::CStateRestoreTraverser& traverser) {
    std::size_t numResults{0};

    do {
        const std::string& name = traverser.name();
        if (name == CONSTANT_TAG) {
            doRestore<CConstantPrior>(ptr, traverser);
            ++numResults;
        } else if (name == GAMMA_TAG) {
            doRestore<CGammaRateConjugate>(params, ptr, traverser);
            ++numResults;
        } else if (name == LOG_NORMAL_TAG) {
            doRestore<CLogNormalMeanPrecConjugate>(params, ptr, traverser);
            ++numResults;
        } else if (name == MULTIMODAL_TAG) {
            doRestore<CMultimodalPrior>(params, ptr, traverser);
            ++numResults;
        } else if (name == MULTINOMIAL_TAG) {
            doRestore<CMultinomialConjugate>(params, ptr, traverser);
            ++numResults;
        } else if (name == NORMAL_TAG) {
            doRestore<CNormalMeanPrecConjugate>(params, ptr, traverser);
            ++numResults;
        } else if (name == ONE_OF_N_TAG) {
            doRestore<COneOfNPrior>(params, ptr, traverser);
            ++numResults;
        } else if (name == POISSON_TAG) {
            doRestore<CPoissonMeanConjugate>(params, ptr, traverser);
            ++numResults;
        } else {
            // Due to the way we divide large state into multiple chunks
            // this is not necessarily a problem - the unexpected element may be
            // marking the start of a new chunk
            LOG_WARN(<< "No prior distribution corresponds to node name "
                     << traverser.name());
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR(<< "Expected 1 (got " << numResults << ") prior model tags");
        ptr.reset();
        return false;
    }

    return true;
}
}

bool CPriorStateSerialiser::operator()(const SDistributionRestoreParams& params,
                                       TPriorUPtr& ptr,
                                       core::CStateRestoreTraverser& traverser) const {
    return restore(params, ptr, traverser);
}

bool CPriorStateSerialiser::operator()(const SDistributionRestoreParams& params,
                                       TPriorSPtr& ptr,
                                       core::CStateRestoreTraverser& traverser) const {
    return restore(params, ptr, traverser);
}

void CPriorStateSerialiser::operator()(const CPrior& prior,
                                       core::CStatePersistInserter& inserter) const {
    std::string tagName;

    if (dynamic_cast<const CConstantPrior*>(&prior) != nullptr) {
        tagName = CONSTANT_TAG;
    } else if (dynamic_cast<const CGammaRateConjugate*>(&prior) != nullptr) {
        tagName = GAMMA_TAG;
    } else if (dynamic_cast<const CLogNormalMeanPrecConjugate*>(&prior) != nullptr) {
        tagName = LOG_NORMAL_TAG;
    } else if (dynamic_cast<const CMultimodalPrior*>(&prior) != nullptr) {
        tagName = MULTIMODAL_TAG;
    } else if (dynamic_cast<const CMultinomialConjugate*>(&prior) != nullptr) {
        tagName = MULTINOMIAL_TAG;
    } else if (dynamic_cast<const CNormalMeanPrecConjugate*>(&prior) != nullptr) {
        tagName = NORMAL_TAG;
    } else if (dynamic_cast<const COneOfNPrior*>(&prior) != nullptr) {
        tagName = ONE_OF_N_TAG;
    } else if (dynamic_cast<const CPoissonMeanConjugate*>(&prior) != nullptr) {
        tagName = POISSON_TAG;
    } else {
        LOG_ERROR(<< "Prior distribution with type '" << typeid(prior).name()
                  << "' has no defined field name");
        return;
    }

    inserter.insertLevel(tagName, std::bind(&CPrior::acceptPersistInserter,
                                            &prior, std::placeholders::_1));
}

bool CPriorStateSerialiser::operator()(const SDistributionRestoreParams& params,
                                       TMultivariatePriorPtr& ptr,
                                       core::CStateRestoreTraverser& traverser) const {
    std::size_t numResults = 0u;

    do {
        const std::string& name = traverser.name();
        if (name == CMultivariatePrior::CONSTANT_TAG) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(
                    name.substr(CMultivariatePrior::CONSTANT_TAG.length()), dimension) == false) {
                LOG_ERROR(<< "Bad dimension encoded in " << name);
                return false;
            }
            ptr.reset(new CMultivariateConstantPrior(dimension, traverser));
            ++numResults;
        } else if (name.find(CMultivariatePrior::MULTIMODAL_TAG) != std::string::npos) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(
                    name.substr(CMultivariatePrior::MULTIMODAL_TAG.length()),
                    dimension) == false) {
                LOG_ERROR(<< "Bad dimension encoded in " << name);
                return false;
            }
            CMultivariateMultimodalPriorFactory::restore(dimension, params, ptr, traverser);
            ++numResults;
        } else if (name.find(CMultivariatePrior::NORMAL_TAG) != std::string::npos) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(
                    name.substr(CMultivariatePrior::NORMAL_TAG.length()), dimension) == false) {
                LOG_ERROR(<< "Bad dimension encoded in " << name);
                return false;
            }
            CMultivariateNormalConjugateFactory::restore(dimension, params, ptr, traverser);
            ++numResults;
        } else if (name.find(CMultivariatePrior::ONE_OF_N_TAG) != std::string::npos) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(
                    name.substr(CMultivariatePrior::ONE_OF_N_TAG.length()), dimension) == false) {
                LOG_ERROR(<< "Bad dimension encoded in " << name);
                return false;
            }
            CMultivariateOneOfNPriorFactory::restore(dimension, params, ptr, traverser);
            ++numResults;
        } else {
            // Due to the way we divide large state into multiple chunks
            // this is not necessarily a problem - the unexpected element may be
            // marking the start of a new chunk
            LOG_WARN(<< "No prior distribution corresponds to node name "
                     << traverser.name());
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR(<< "Expected 1 (got " << numResults << ") prior model tags");
        ptr.reset();
        return false;
    }

    return true;
}

void CPriorStateSerialiser::operator()(const CMultivariatePrior& prior,
                                       core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(prior.persistenceTag(),
                         std::bind(&CMultivariatePrior::acceptPersistInserter,
                                   &prior, std::placeholders::_1));
}
}
}
