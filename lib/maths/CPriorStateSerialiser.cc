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

#include <boost/bind.hpp>

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
}

bool CPriorStateSerialiser::operator()(const SDistributionRestoreParams &params,
                                       TPriorPtr &ptr,
                                       core::CStateRestoreTraverser &traverser) const {
    size_t numResults(0);

    do {
        const std::string &name = traverser.name();
        if (name == CONSTANT_TAG) {
            ptr.reset(new CConstantPrior(traverser));
            ++numResults;
        } else if (name == GAMMA_TAG) {
            ptr.reset(new CGammaRateConjugate(params, traverser));
            ++numResults;
        } else if (name == LOG_NORMAL_TAG) {
            ptr.reset(new CLogNormalMeanPrecConjugate(params, traverser));
            ++numResults;
        } else if (name == MULTIMODAL_TAG) {
            ptr.reset(new CMultimodalPrior(params, traverser));
            ++numResults;
        } else if (name == MULTINOMIAL_TAG) {
            ptr.reset(new CMultinomialConjugate(params, traverser));
            ++numResults;
        } else if (name == NORMAL_TAG) {
            ptr.reset(new CNormalMeanPrecConjugate(params, traverser));
            ++numResults;
        } else if (name == ONE_OF_N_TAG) {
            ptr.reset(new COneOfNPrior(params, traverser));
            ++numResults;
        } else if (name == POISSON_TAG) {
            ptr.reset(new CPoissonMeanConjugate(params, traverser));
            ++numResults;
        } else {
            // Due to the way we divide large state into multiple chunks
            // this is not necessarily a problem - the unexpected element may be
            // marking the start of a new chunk
            LOG_WARN("No prior distribution corresponds to node name " << traverser.name());
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR("Expected 1 (got " << numResults << ") prior model tags");
        ptr.reset();
        return false;
    }

    return true;
}

void CPriorStateSerialiser::operator()(const CPrior &prior,
                                       core::CStatePersistInserter &inserter) const {
    std::string tagName;

    if (dynamic_cast<const CConstantPrior *>(&prior) != 0) {
        tagName = CONSTANT_TAG;
    } else if (dynamic_cast<const CGammaRateConjugate *>(&prior) != 0) {
        tagName = GAMMA_TAG;
    } else if (dynamic_cast<const CLogNormalMeanPrecConjugate *>(&prior) != 0) {
        tagName = LOG_NORMAL_TAG;
    } else if (dynamic_cast<const CMultimodalPrior *>(&prior) != 0) {
        tagName = MULTIMODAL_TAG;
    } else if (dynamic_cast<const CMultinomialConjugate *>(&prior) != 0) {
        tagName = MULTINOMIAL_TAG;
    } else if (dynamic_cast<const CNormalMeanPrecConjugate *>(&prior) != 0) {
        tagName = NORMAL_TAG;
    } else if (dynamic_cast<const COneOfNPrior *>(&prior) != 0) {
        tagName = ONE_OF_N_TAG;
    } else if (dynamic_cast<const CPoissonMeanConjugate *>(&prior) != 0) {
        tagName = POISSON_TAG;
    } else {
        LOG_ERROR("Prior distribution with type '" << typeid(prior).name()
                                                   << "' has no defined field name");
        return;
    }

    inserter.insertLevel(tagName, boost::bind(&CPrior::acceptPersistInserter, &prior, _1));
}

bool CPriorStateSerialiser::operator()(const SDistributionRestoreParams &params,
                                       TMultivariatePriorPtr &ptr,
                                       core::CStateRestoreTraverser &traverser) const {
    std::size_t numResults = 0u;

    do {
        const std::string &name = traverser.name();
        if (name == CMultivariatePrior::CONSTANT_TAG) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(name.substr(CMultivariatePrior::CONSTANT_TAG.length()), dimension) == false) {
                LOG_ERROR("Bad dimension encoded in " << name);
                return false;
            }
            ptr.reset(new CMultivariateConstantPrior(dimension, traverser));
            ++numResults;
        } else if (name.find(CMultivariatePrior::MULTIMODAL_TAG) != std::string::npos) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(name.substr(CMultivariatePrior::MULTIMODAL_TAG.length()), dimension) == false) {
                LOG_ERROR("Bad dimension encoded in " << name);
                return false;
            }
            CMultivariateMultimodalPriorFactory::restore(dimension, params, ptr, traverser);
            ++numResults;
        } else if (name.find(CMultivariatePrior::NORMAL_TAG) != std::string::npos) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(name.substr(CMultivariatePrior::NORMAL_TAG.length()), dimension) == false) {
                LOG_ERROR("Bad dimension encoded in " << name);
                return false;
            }
            CMultivariateNormalConjugateFactory::restore(dimension, params, ptr, traverser);
            ++numResults;
        } else if (name.find(CMultivariatePrior::ONE_OF_N_TAG) != std::string::npos) {
            std::size_t dimension;
            if (core::CStringUtils::stringToType(name.substr(CMultivariatePrior::ONE_OF_N_TAG.length()), dimension) == false) {
                LOG_ERROR("Bad dimension encoded in " << name);
                return false;
            }
            CMultivariateOneOfNPriorFactory::restore(dimension, params, ptr, traverser);
            ++numResults;
        } else {
            // Due to the way we divide large state into multiple chunks
            // this is not necessarily a problem - the unexpected element may be
            // marking the start of a new chunk
            LOG_WARN("No prior distribution corresponds to node name " << traverser.name());
        }
    } while (traverser.next());

    if (numResults != 1) {
        LOG_ERROR("Expected 1 (got " << numResults << ") prior model tags");
        ptr.reset();
        return false;
    }

    return true;
}

void CPriorStateSerialiser::operator()(const CMultivariatePrior &prior,
                                       core::CStatePersistInserter &inserter) const {
    inserter.insertLevel(prior.persistenceTag(),
                         boost::bind(&CMultivariatePrior::acceptPersistInserter, &prior, _1));
}

}
}

