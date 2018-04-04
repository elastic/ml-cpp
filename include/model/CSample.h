/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CSample_h
#define INCLUDED_ml_model_CSample_h

#include <core/CMemoryUsage.h>
#include <core/CoreTypes.h>
#include <core/CSmallVector.h>

#include <model/ImportExport.h>

#include <cstddef>
#include <string>

namespace ml
{
namespace model
{

//! \brief A metric sample statistic.
class MODEL_EXPORT CSample
{
    public:
        using TDouble1Vec = core::CSmallVector<double, 1>;

        struct MODEL_EXPORT SToString
        {
            std::string operator()(const CSample &sample) const;
        };

        struct MODEL_EXPORT SFromString
        {
            bool operator()(const std::string &token, CSample &value) const;
        };

    public:
        CSample();
        CSample(core_t::TTime time, const TDouble1Vec &value, double varianceScale, double count);

        //! Get the time.
        core_t::TTime time() const { return m_Time; }

        //! Get the variance scale.
        double varianceScale() const { return m_VarianceScale; }

        //! Get the count.
        double count() const { return m_Count; }

        //! Get a writable count.
        double &count() { return m_Count; }

        //! Get the value and any ancillary statistics needed to calculate
        //! influence.
        const TDouble1Vec &value() const { return m_Value; }

        //! Get a writable value and any ancillary statistics needed to
        //! calculate influence.
        TDouble1Vec &value() { return m_Value; }

        //! Get the value of the feature.
        TDouble1Vec value(std::size_t dimension) const;

        //! Get a checksum.
        uint64_t checksum() const;

        //! Print the sample for debug.
        std::string print() const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        core_t::TTime m_Time;
        TDouble1Vec m_Value;
        double m_VarianceScale;
        double m_Count;
};

}
}

#endif // INCLUDED_ml_model_CSample_h
