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
        CSample(void);
        CSample(core_t::TTime time, const TDouble1Vec &value, double varianceScale, double count);

        //! Get the time.
        core_t::TTime time(void) const { return m_Time; }

        //! Get the variance scale.
        double varianceScale(void) const { return m_VarianceScale; }

        //! Get the count.
        double count(void) const { return m_Count; }

        //! Get a writable count.
        double &count(void) { return m_Count; }

        //! Get the value and any ancillary statistics needed to calculate
        //! influence.
        const TDouble1Vec &value(void) const { return m_Value; }

        //! Get a writable value and any ancillary statistics needed to
        //! calculate influence.
        TDouble1Vec &value(void) { return m_Value; }

        //! Get the value of the feature.
        TDouble1Vec value(std::size_t dimension) const;

        //! Get a checksum.
        uint64_t checksum(void) const;

        //! Print the sample for debug.
        std::string print(void) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage(void) const;

    private:
        core_t::TTime m_Time;
        TDouble1Vec m_Value;
        double m_VarianceScale;
        double m_Count;
};

}
}

#endif // INCLUDED_ml_model_CSample_h
