/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CAdaptiveBucketing_h
#define INCLUDED_ml_maths_CAdaptiveBucketing_h

#include <core/CMemory.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CSpline.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStateRestoreTraverser;
class CStatePersistInserter;
}
namespace maths {

//! \brief Common functionality used by our adaptive bucketing classes.
//!
//! DESCRIPTION:\n
//! This implements an adaptive bucketing strategy for the points
//! of a periodic function. The idea is to adjust the bucket end
//! points to efficiently capture the function detail in a fixed
//! number of buckets. Function values are assumed to have additive
//! noise. In particular, it is assumed that the observed values
//! \f$(x_i, y_i)\f$ are described by:
//! <pre class="fragment">
//!   \f$\{(x_i, y_i = y(x_i) + Y_i\}\f$
//! </pre>
//!
//! Here, \f$Y_i\f$ are IID mean zero random variables. We are
//! interested in spacing the buckets to minimize the maximum
//! error in approximating the function by its mean in each bucket,
//! i.e. we'd like to minimize:
//! <pre class="fragment">
//!   \f$\displaystyle \max_i\left\{ \int_{[a_i,b_i]}{ \left| y(x) - \left<y\right>_{[a_i,b_i]} \right| }dx \right\} \f$
//! </pre>
//!
//! Here, \f$\left<y\right>_{[a_i,b_i]} = \frac{1}{b_i-a_i}\int_{[a_i,b_i]}{y(x)}dx\f$.
//! It is relatively straightforward to show that if the points are
//! uniformly distributed in the function domain then the mean in
//! each bucket is a unbiased estimator of \f$\left<y\right>\f$ in that
//! bucket. We estimate the error by using the mean smoothed central
//! range of the function in each bucket, given by difference between
//! adjacent function bucket means. The smoothing is achieved by
//! convolution. (This empirically gives better results for smooth
//! functions and is also beneficial for spline interpolation where
//! it is desirable to increase the number of knots _near_ regions
//! of high curvature to control the function.)
//!
//! For sufficiently smooth functions and a given number of buckets
//! the objective is minimized by ensuring that "bucket width" x
//! "function range" is equal in all buckets.
//!
//! The bucketing is aged by relaxing it back towards uniform.
class MATHS_EXPORT CAdaptiveBucketing {
public:
    using TDoubleVec = std::vector<double>;
    using TFloatVec = std::vector<CFloatStorage>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;

public:
    //! Refine the bucket end points to minimize the maximum averaging
    //! error in any bucket.
    //!
    //! \param[in] time The time at which to refine.
    void refine(core_t::TTime time);

    //! Check if the bucketing has been initialized.
    bool initialized() const;

    //! Get the number of buckets.
    std::size_t size() const;

    //! Set the rate at which the bucketing loses information.
    void decayRate(double value);

    //! Get the rate at which the bucketing loses information.
    double decayRate() const;

    //! Get the minimum permitted bucket length.
    double minimumBucketLength() const;

    //! Get the bucket end points.
    const TFloatVec& endpoints() const;

    //! Get the bucket value centres.
    const TFloatVec& centres() const;

    //! Get the bucket value centres.
    const TFloatVec& largeErrorCounts() const;

    //! Get a set of knot points and knot point values to use for
    //! interpolating the bucket values.
    //!
    //! \param[in] time The time at which to get the knot points.
    //! \param[in] boundary Controls the style of start and end knots.
    //! \param[out] knots Filled in with the knot points to interpolate.
    //! \param[out] values Filled in with the values at \p knots.
    //! \param[out] variances Filled in with the variances at \p knots.
    //! \return True if there are sufficient knot points to interpolate
    //! and false otherwise.
    bool knots(core_t::TTime time,
               CSplineTypes::EBoundaryCondition boundary,
               TDoubleVec& knots,
               TDoubleVec& values,
               TDoubleVec& variances) const;

    //! \name Test Functions
    //@{
    //! Get the total count of in the bucketing.
    double count() const;

    //! Get the bucket regressions.
    TDoubleVec values(core_t::TTime time) const;

    //! Get the bucket variances.
    TDoubleVec variances() const;
    //@}

    //! Name of component
    virtual std::string name() const = 0;

    //! Check that the state is valid.
    virtual bool isBad() const = 0;

    //! \brief Ensure \p points are at least \p separation apart.
    //!
    //! This solves the problem of finding new positions for \p points
    //! such that no pair of points is closer than \p separation and
    //! the total square distance they move is minimized.
    //!
    //! \param[in] a The left end of the interval that must contain \p points.
    //! \param[in] b The right end of the interval that must contain \p points.
    //! \param[in] separation The minimum permitted distance between \p points.
    //! \param[in,out] points The points to spread.
    static void spread(double a, double b, double separation, TFloatVec& points);

protected:
    using TRestoreFunc = std::function<bool(core::CStateRestoreTraverser&)>;
    using TPersistFunc = std::function<void(core::CStatePersistInserter&)>;

protected:
    //! The minimum number of standard deviations for an error to be
    //! considered large.
    static const double LARGE_ERROR_STANDARD_DEVIATIONS;

protected:
    CAdaptiveBucketing(double decayRate, double minimumBucketLength);
    virtual ~CAdaptiveBucketing() = default;

    //! Get the restore function bound to this object.
    TRestoreFunc getAcceptRestoreTraverser();

    //! Get the accept persist function bound to this object.
    TPersistFunc getAcceptPersistInserter() const;

    //! Check the state invariants after restoration
    //! Abort on failure.
    void checkRestoredInvariants() const;

    //! Restore by traversing a state document
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Efficiently swap the contents of two bucketing objects.
    void swap(CAdaptiveBucketing& other) noexcept;

    //! Create a new uniform bucketing with \p n buckets on the
    //! interval [\p a, \p b].
    //!
    //! \param[in] a The start of the interval to bucket.
    //! \param[in] b The end of the interval to bucket.
    //! \param[in] n The number of buckets.
    bool initialize(double a, double b, std::size_t n);

    //! Add the function mean values \f$([a_i,b_i], m_i)\f$ where
    //! \f$m_i\f$ are the means of the function in the time intervals
    //! \f$([a+(i-1)l,b+il])\f$, \f$i\in[n]\f$ and \f$l=(b-a)/n\f$.
    //!
    //! \param[in] startTime The start of the period.
    //! \param[in] endTime The start of the period.
    //! \param[in] values The mean values in a regular subdivision
    //! of [\p start,\p end].
    void initialValues(core_t::TTime startTime,
                       core_t::TTime endTime,
                       const TFloatMeanAccumulatorVec& values);

    //! Clear the contents of this bucketing and recover any
    //! allocated memory.
    void clear();

    //! Add the function value at \p time.
    //!
    //! \param[in] bucket The index of the bucket of \p time.
    //! \param[in] time The time of the value being added.
    //! \param[in] weight The weight of the value being added. The
    //! smaller this is the less influence it has on the bucket.
    void add(std::size_t bucket, core_t::TTime time, double weight);

    //! Add a large error in \p bucket.
    void addLargeError(std::size_t bucket, core_t::TTime time);

    //! Age the force moments.
    void age(double factor);

    //! Get the bucket value centres.
    TFloatVec& centres();

    //! Get the bucket value centres.
    TFloatVec& largeErrorCounts();

    //! Adjust \p weight for significant large error counts.
    double adjustedWeight(std::size_t bucket, double weight) const;

    //! Compute the index of the bucket to which \p time belongs
    bool bucket(core_t::TTime time, std::size_t& result) const;

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Get the memory used by this component
    std::size_t memoryUsage() const;

private:
    using TFloatUInt32Pr = std::pair<CFloatStorage, std::uint32_t>;
    using TFloatUInt32PrMinAccumulator = CBasicStatistics::SMin<TFloatUInt32Pr, 2>::TAccumulator;

private:
    //! Compute the values corresponding to the change in end
    //! points from \p endpoints. The values are assigned based
    //! on their intersection with each bucket in the previous
    //! bucket configuration.
    virtual void refresh(const TFloatVec& endpoints) = 0;

    //! Check if \p time is in the this component's window.
    virtual bool inWindow(core_t::TTime time) const = 0;

    //! Add the function initial value at \p time.
    virtual void
    addInitialValue(std::size_t bucket, core_t::TTime time, double value, double weight) = 0;

    //! Get the offset w.r.t. the start of the bucketing of \p time.
    virtual double offset(core_t::TTime time) const = 0;

    //! Get the count in \p bucket.
    virtual double bucketCount(std::size_t bucket) const = 0;

    //! Get the predicted value for \p bucket at \p time.
    virtual double predict(std::size_t bucket, core_t::TTime time, double offset) const = 0;

    //! Get the variance of \p bucket.
    virtual double variance(std::size_t bucket) const = 0;

    //! Implements split of \p bucket for derived state.
    virtual void split(std::size_t bucket) = 0;

    //! Compute the p-value of the large error count for \p bucket.
    double bucketLargeErrorCountPValue(double totalLargeErrorCount, std::size_t bucket) const;

    //! Check if there is evidence of systematically large errors in a
    //! bucket and split it if there is.
    void maybeSplitBucket();

    //! Split \p bucket.
    void splitBucket(std::size_t bucket);

private:
    //! The rate at which information is aged out of the bucket values.
    double m_DecayRate;

    //! The minimum permitted bucket length if non-zero otherwise this
    //! is ignored.
    double m_MinimumBucketLength;

    //! The desired number of buckets. We can use more if we determine
    //! that we aren't capturing the periodic pattern effectively.
    //!
    //! \see maybeSplitBucketMostSignificantBuckets for details.
    std::size_t m_TargetSize = 0;

    //! The bucket of the last large error added.
    std::size_t m_LastLargeErrorBucket = 0;

    //! The period of the last large error added.
    core_t::TTime m_LastLargeErrorPeriod = 0;

    //! The p-values of the most significant large error counts.
    TFloatUInt32PrMinAccumulator m_LargeErrorCountPValues;

    //! The mean weight of values added.
    TFloatMeanAccumulator m_MeanWeight;

    //! The bucket end points.
    TFloatVec m_Endpoints;

    //! The mean offset (relative to the start of the bucket) of samples
    //! in each bucket.
    TFloatVec m_Centres;

    //! The count of large errors in each bucket.
    TFloatVec m_LargeErrorCounts;

    //! An IIR low pass filter for the total desired end point displacement
    //! in refine.
    TFloatMeanAccumulator m_MeanDesiredDisplacement;

    //! The total desired end point displacement in refine.
    TFloatMeanAccumulator m_MeanAbsDesiredDisplacement;
};
}
}

#endif // INCLUDED_ml_maths_CAdaptiveBucketing_h
