/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CMetricStatisticWrappers_h
#define INCLUDED_ml_model_CMetricStatisticWrappers_h

#include <core/CSmallVector.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBasicStatisticsPersist.h>
#include <maths/CDoublePrecisionStorage.h>
#include <maths/CQuantileSketch.h>

#include <model/ImportExport.h>

#include <boost/bind.hpp>

#include <vector>

namespace ml
{
namespace model
{
template<typename STATISTIC> class CMetricMultivariateStatistic;

namespace metric_statistic_wrapper_detail
{

//! \brief Makes a univariate metric statistic.
template<typename STATISTIC>
struct SMake
{
    static STATISTIC dispatch(std::size_t /*dimension*/)
    {
        return STATISTIC();
    }
};
//! \brief Makes a multivariate metric statistic.
template<typename STATISTIC>
struct SMake<CMetricMultivariateStatistic<STATISTIC> >
{
    static CMetricMultivariateStatistic<STATISTIC> dispatch(std::size_t dimension)
    {
        return CMetricMultivariateStatistic<STATISTIC>(dimension);
    }

};

} // metric_statistic_wrapper_detail::

//! \brief Provides wrappers for all aggregate metric statistics
//! for which we gather data.
//!
//! DESCTIPTION:\n
//! This shim is used by CPartialStatistic and CSampleGatherer to
//! provide a common interface into the various types of operation
//! which those classes need.
//!
//! It provides static functions for getting the statistic value
//! and count if possible, and persisting and restoring them all
//! of which delegate to the appropriate statistic functions.
struct MODEL_EXPORT CMetricStatisticWrappers
{
    typedef core::CSmallVector<double, 1> TDouble1Vec;
    typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
    typedef maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator TVarianceAccumulator;
    typedef maths::CFixedQuantileSketch<maths::CQuantileSketch::E_PiecewiseConstant, 30> TMedianAccumulator;

    //! Make a statistic.
    template<typename STATISTIC>
    static STATISTIC make(std::size_t dimension)
    {
        return metric_statistic_wrapper_detail::SMake<STATISTIC>::dispatch(dimension);
    }

    //! Add \p value to an order statistic.
    template<typename LESS>
    static void add(const TDouble1Vec &value,
                    unsigned int count,
                    maths::CBasicStatistics::COrderStatisticsStack<double, 1, LESS> &stat)
    {
        stat.add(value[0], count);
    }
    //! Add \p value to a mean statistic.
    static void add(const TDouble1Vec &value, unsigned int count, TMeanAccumulator &stat)
    {
        stat.add(value[0], count);
    }
    //! Add \p value to a variance statistic.
    static void add(const TDouble1Vec &value, unsigned int count, TVarianceAccumulator &stat)
    {
        stat.add(value[0], count);
    }
    //! Add \p value to a median statistic.
    static void add(const TDouble1Vec &value, unsigned int count, TMedianAccumulator &stat)
    {
        stat.add(value[0], count);
    }
    //! Add \p value to a multivariate statistic.
    template<typename STATISTIC>
    static void add(const TDouble1Vec &value,
                    unsigned int count,
                    CMetricMultivariateStatistic<STATISTIC> &stat)
    {
        stat.add(value, count);
    }

    //! Get the median value of an order statistic.
    template<typename LESS>
    static TDouble1Vec value(const maths::CBasicStatistics::COrderStatisticsStack<double, 1, LESS> & stat)
    {
        return TDouble1Vec{stat[0]};
    }
    //! Get the value of a mean statistic.
    static TDouble1Vec value(const TMeanAccumulator &stat)
    {
        return TDouble1Vec{maths::CBasicStatistics::mean(stat)};
    }
    //! Get the value of a variance statistic.
    static TDouble1Vec value(const TVarianceAccumulator &stat)
    {
        TDouble1Vec result;
        if (maths::CBasicStatistics::count(stat) >= 2.0)
        {
            result.assign({maths::CBasicStatistics::maximumLikelihoodVariance(stat),
                           maths::CBasicStatistics::mean(stat)});
        }
        return result;
    }
    //! Get the value of a median statistic.
    static TDouble1Vec value(const TMedianAccumulator &stat)
    {
        double result;
        if (!stat.quantile(50.0, result))
        {
            return TDouble1Vec{0.0};
        }
        return TDouble1Vec{result};
    }
    //! Get the value of a multivariate statistic.
    template<typename STATISTIC>
    static TDouble1Vec value(const CMetricMultivariateStatistic<STATISTIC> &stat)
    {
        return stat.value();
    }

    //! Forward to the value function.
    template<typename STATISTIC>
    static TDouble1Vec influencerValue(const STATISTIC &stat)
    {
        return value(stat);
    }
    //! Get the variance influence value.
    static TDouble1Vec influencerValue(const TVarianceAccumulator &stat)
    {
        // We always return an influence value (independent of the count)
        // because this is not used to directly compute a variance only
        // to adjust the bucket variance.
        TDouble1Vec result(2);
        result[0] = maths::CBasicStatistics::maximumLikelihoodVariance(stat);
        result[1] = maths::CBasicStatistics::mean(stat);
        return result;
    }
    //! Get the value suitable for computing influence of a multivariate
    //! statistic.
    template<typename STATISTIC>
    static TDouble1Vec influencerValue(const CMetricMultivariateStatistic<STATISTIC> &stat)
    {
        return stat.influencerValue();
    }

    //! Returns 1.0 since this is not available.
    template<typename LESS>
    static double count(const maths::CBasicStatistics::COrderStatisticsStack<double, 1, LESS> &/*stat*/)
    {
        return 1.0;
    }
    //! Get the count of the statistic.
    static double count(const TMeanAccumulator &stat)
    {
        return static_cast<double>(maths::CBasicStatistics::count(stat));
    }
    //! Get the count of the statistic.
    static double count(const TVarianceAccumulator &stat)
    {
        return static_cast<double>(maths::CBasicStatistics::count(stat));
    }
    //! Get the count of the statistic.
    static double count(const TMedianAccumulator &stat)
    {
        return stat.count();
    }
    //! Get the count of a multivariate statistic.
    template<typename STATISTIC>
    static double count(const CMetricMultivariateStatistic<STATISTIC> &stat)
    {
        return stat.count();
    }

    //! Persist an order statistic.
    template<typename LESS>
    static void persist(const maths::CBasicStatistics::COrderStatisticsStack<double, 1, LESS> &stat,
                        const std::string &tag,
                        core::CStatePersistInserter &inserter)
    {
        inserter.insertValue(tag, stat.toDelimited());
    }
    //! Persist a mean statistic.
    static void persist(const TMeanAccumulator &stat,
                        const std::string &tag,
                        core::CStatePersistInserter &inserter)
    {
        inserter.insertValue(tag, stat.toDelimited());
    }
    //! Persist a variance statistic.
    static void persist(const TVarianceAccumulator &stat,
                        const std::string &tag,
                        core::CStatePersistInserter &inserter)
    {
        inserter.insertValue(tag, stat.toDelimited());
    }
    //! Persist a median statistic.
    static void persist(const TMedianAccumulator &stat,
                        const std::string &tag,
                        core::CStatePersistInserter &inserter)
    {
        inserter.insertLevel(tag, boost::bind(&TMedianAccumulator::acceptPersistInserter, &stat, _1));
    }
    //! Persist a multivariate statistic.
    template<typename STATISTIC>
    static void persist(const CMetricMultivariateStatistic<STATISTIC> &stat,
                        const std::string &tag,
                        core::CStatePersistInserter &inserter)
    {
        inserter.insertLevel(tag, boost::bind(&CMetricMultivariateStatistic<STATISTIC>::persist, &stat, _1));
    }

    //! Restore an order statistic.
    template<typename LESS>
    static inline bool restore(core::CStateRestoreTraverser &traverser,
                               maths::CBasicStatistics::COrderStatisticsStack<double, 1, LESS> &stat)
    {
        if (stat.fromDelimited(traverser.value()) == false)
        {
            LOG_ERROR("Invalid statistic in " << traverser.value());
            return false;
        }
        return true;
    }
    //! Restore a mean statistic.
    static bool restore(core::CStateRestoreTraverser &traverser, TMeanAccumulator &stat)
    {
        if (stat.fromDelimited(traverser.value()) == false)
        {
            LOG_ERROR("Invalid mean in " << traverser.value());
            return false;
        }
        return true;
    }
    //! Restore a variance statistic.
    static bool restore(core::CStateRestoreTraverser &traverser, TVarianceAccumulator &stat)
    {
        if (stat.fromDelimited(traverser.value()) == false)
        {
            LOG_ERROR("Invalid variance in " << traverser.value());
            return false;
        }
        return true;
    }
    //! Restore a median statistic.
    static bool restore(core::CStateRestoreTraverser &traverser, TMedianAccumulator &stat)
    {
        return traverser.traverseSubLevel(boost::bind(&TMedianAccumulator::acceptRestoreTraverser, &stat, _1));
    }
    //! Restore a multivariate statistic.
    template<typename STATISTIC>
    static bool restore(core::CStateRestoreTraverser &traverser,
                        CMetricMultivariateStatistic<STATISTIC> &stat)
    {
        return traverser.traverseSubLevel(boost::bind(&CMetricMultivariateStatistic<STATISTIC>::restore, &stat, _1));
    }
};

}
}

#endif // INCLUDED_ml_model_CMetricStatisticWrappers_h
