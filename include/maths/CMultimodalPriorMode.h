/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_SMultimodalPriorMode_h
#define INCLUDED_ml_maths_SMultimodalPriorMode_h

#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CPriorStateSerialiser.h>

#include <cstddef>
#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

namespace ml {
namespace maths {

//! \brief The prior of a mode of the likelihood function and
//! a unique identifier for the clusterer.
//!
//! DESCRIPTION:\n
//! See, for example, CMultimodalPrior for usage.
template<typename PRIOR_PTR>
struct SMultimodalPriorMode {
    static const core::TPersistenceTag INDEX_TAG;
    static const core::TPersistenceTag PRIOR_TAG;

    SMultimodalPriorMode() : s_Index(0), s_Prior() {}
    SMultimodalPriorMode(std::size_t index, const PRIOR_PTR& prior)
        : s_Index(index), s_Prior(prior->clone()) {}
    SMultimodalPriorMode(std::size_t index, PRIOR_PTR&& prior)
        : s_Index(index), s_Prior(std::move(prior)) {}

    //! Get the weight of this sample.
    double weight() const { return s_Prior->numberSamples(); }

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed) const {
        seed = CChecksum::calculate(seed, s_Index);
        return CChecksum::calculate(seed, s_Prior);
    }

    //! Get the memory used by this component
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const {
        mem->setName("CMultimodalPrior::SMode");
        core::CMemoryDebug::dynamicSize("s_Prior", s_Prior, mem);
    }

    //! Get the memory used by this component
    std::size_t memoryUsage() const {
        return core::CMemory::dynamicSize(s_Prior);
    }

    //! Create from part of a state document.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser) {
        do {
            const std::string& name = traverser.name();
            RESTORE_BUILT_IN(INDEX_TAG, s_Index)
            RESTORE(PRIOR_TAG, traverser.traverseSubLevel(std::bind<bool>(
                                   CPriorStateSerialiser(), std::cref(params),
                                   std::ref(s_Prior), std::placeholders::_1)))
        } while (traverser.next());

        return true;
    }

    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const {
        inserter.insertValue(INDEX_TAG, s_Index);
        inserter.insertLevel(PRIOR_TAG, std::bind<void>(CPriorStateSerialiser(),
                                                        std::cref(*s_Prior),
                                                        std::placeholders::_1));
    }

    //! Full debug dump of the mode weights.
    template<typename T>
    static std::string debugWeights(const std::vector<SMultimodalPriorMode<T>>& modes) {
        if (modes.empty()) {
            return std::string();
        }
        std::ostringstream result;
        result << std::scientific << std::setprecision(15) << modes[0].weight();
        for (std::size_t i = 1; i < modes.size(); ++i) {
            result << " " << modes[i].weight();
        }
        return result.str();
    }

    std::size_t s_Index;
    PRIOR_PTR s_Prior;
};

template<typename PRIOR>
const core::TPersistenceTag SMultimodalPriorMode<PRIOR>::INDEX_TAG("a", "index");
template<typename PRIOR>
const core::TPersistenceTag SMultimodalPriorMode<PRIOR>::PRIOR_TAG("b", "prior");
}
}

#endif // INCLUDED_ml_maths_SMultimodalPriorMode_h
