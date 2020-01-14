/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CMonitoredResource_h
#define INCLUDED_ml_model_CMonitoredResource_h

#include <core/CMemoryUsage.h>
#include <core/CoreTypes.h>

#include <model/CResourceMonitor.h>
#include <model/ImportExport.h>

namespace ml {
namespace model {

//! \brief
//! Interface for resource monitoring.
//!
//! DESCRIPTION:\n
//! Resources that the CResourceMonitor class monitors must
//! derive from this interface.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Defaults to the assumption that monitored resources do not support
//! pruning.  Methods related to pruning have default implementations
//! consistent with this assumption.
//!
class MODEL_EXPORT CMonitoredResource {
public:
    virtual ~CMonitoredResource() = default;

    //! Get the memory used by this detector.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Return the total memory usage.
    virtual std::size_t memoryUsage() const = 0;

    //! Get the static size of the derived class
    virtual std::size_t staticSize() const = 0;

    //! Does this monitored resource support pruning?
    virtual bool supportsPruning() const;

    //! Try to initialize the pruning window.
    //! \return true if the pruning window was successfully initialized,
    //!         otherwise false.
    virtual bool initPruneWindow(std::size_t& defaultPruneWindow,
                                 std::size_t& minimumPruneWindow) const;

    //! Get the bucket length appropriate for adjusting the pruning window.
    //! The return value is 0 if this monitored resource does not support
    //! pruning.
    virtual core_t::TTime bucketLength() const;

    //! Prune this monitored resource, i.e. reduce its memory usage by
    //! discarding the least recently seen entities it knows about.
    virtual void prune(std::size_t maximumAge);

    //! Update the overall model memory stats results with stats from this
    //! monitored resource.
    virtual void updateMemoryResults(CResourceMonitor::SResults& results) const = 0;
};
}
}

#endif // INCLUDED_ml_model_CMonitoredResource_h
