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

#ifndef INCLUDED_ml_maths_SMultimodalPriorMode_h
#define INCLUDED_ml_maths_SMultimodalPriorMode_h

#include <core/CMemory.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>
#include <maths/CPriorStateSerialiser.h>

#include <cstddef>
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
    static const std::string INDEX_TAG;
    static const std::string PRIOR_TAG;

    SMultimodalPriorMode(void) : s_Index(0), s_Prior() {}
    SMultimodalPriorMode(std::size_t index, const PRIOR_PTR &prior) :
        s_Index(index),
        s_Prior(prior->clone())
    {}

    //! Get the weight of this sample.
    double weight(void) const {
        return s_Prior->numberSamples();
    }

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed) const {
        seed = CChecksum::calculate(seed, s_Index);
        return CChecksum::calculate(seed, s_Prior);
    }

    //! Get the memory used by this component
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const {
        mem->setName("CMultimodalPrior::SMode");
        core::CMemoryDebug::dynamicSize("s_Prior", s_Prior, mem);
    }

    //! Get the memory used by this component
    std::size_t memoryUsage(void) const {
        return core::CMemory::dynamicSize(s_Prior);
    }

    //! Create from part of a state document.
    bool acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                core::CStateRestoreTraverser &traverser) {
        do {
            const std::string &name = traverser.name();
            RESTORE_BUILT_IN(INDEX_TAG, s_Index)
            RESTORE(PRIOR_TAG, traverser.traverseSubLevel(boost::bind<bool>(CPriorStateSerialiser(),
                                                                            boost::cref(params),
                                                                            boost::ref(s_Prior), _1)))
        } while (traverser.next());

        return true;
    }

    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter &inserter) const {
        inserter.insertValue(INDEX_TAG, s_Index);
        inserter.insertLevel(PRIOR_TAG, boost::bind<void>(CPriorStateSerialiser(),
                                                          boost::cref(*s_Prior), _1));
    }

    //! Full debug dump of the mode weights.
    template<typename T>
    static std::string debugWeights(const std::vector<SMultimodalPriorMode<T> > &modes) {
        if (modes.empty()) {
            return std::string();
        }
        std::ostringstream result;
        result << std::scientific << std::setprecision(15) << modes[0].weight();
        for (std::size_t i = 1u; i < modes.size(); ++i) {
            result << " " << modes[i].weight();
        }
        return result.str();
    }

    std::size_t s_Index;
    PRIOR_PTR s_Prior;
};

template<typename PRIOR>
const std::string SMultimodalPriorMode<PRIOR>::INDEX_TAG("a");
template<typename PRIOR>
const std::string SMultimodalPriorMode<PRIOR>::PRIOR_TAG("b");

}
}



#endif // INCLUDED_ml_maths_SMultimodalPriorMode_h
