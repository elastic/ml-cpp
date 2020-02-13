/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CQuantileSketch_h
#define INCLUDED_ml_maths_CQuantileSketch_h

#include <core/CMemoryUsage.h>

#include <maths/CPRNG.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/operators.hpp>

#include <cstddef>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief A sketch suitable for c.d.f. queries on a 1d double valued
//! random variable.
//!
//! DESCRIPTION:\n
//! This is simply a collection of points and counts which sketches,
//! with fixed size, a collection of values supplied to the add function
//! in a way that provides good approximate quantile. Note that this
//! closely resembles Ben-Haim and Tom-Tov's histogram sketch, but with
//! various refinements: in particular, for larger sketches the reduction
//! step is delayed for a number of points which is proportional to the
//! size, a better cost function is used (which minimizes the error
//! introduced into the piecewise constant term c.d.f. by a merge step),
//! a bias correction is applied when interpolating and cost caching is
//! used to accelerate reduction.
//!
//! Note this has none of the theoretical guarantees on maximum error
//! which are available for the q-digest, so if you know the range of the
//! variable up front, that is a safer choice for approximate quantile
//! estimation.
class MATHS_EXPORT CQuantileSketch : private boost::addable<CQuantileSketch> {
public:
    using TFloatFloatPr = std::pair<CFloatStorage, CFloatStorage>;
    using TFloatFloatPrVec = std::vector<TFloatFloatPr>;

    //! The types of interpolation used for computing the quantile.
    enum EInterpolation { E_Linear, E_PiecewiseConstant };

public:
    CQuantileSketch(EInterpolation interpolation, std::size_t size);
    virtual ~CQuantileSketch() = default;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Write a description to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Combine two sketches.
    const CQuantileSketch& operator+=(const CQuantileSketch& rhs);

    //! Define a function operator for use with std:: algorithms.
    void operator()(double x) { this->add(x); }

    //! Add \p x to the sketch.
    void add(double x, double n = 1.0);

    //! Age by scaling the counts.
    void age(double factor);

    //! Get the c.d.f at \p x.
    bool cdf(double x, double& result) const;

    //! Get the minimum value added.
    bool minimum(double& result) const;

    //! Get the maximum value added.
    bool maximum(double& result) const;

    //! Get the estimated median absolute deviation.
    bool mad(double& result) const;

    //! Get the quantile corresponding to \p percentage.
    bool quantile(double percentage, double& result) const;

    //! Get the knot values.
    const TFloatFloatPrVec& knots() const;

    //! Get the total count of points added.
    double count() const;

    //! Get a checksum of this object.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! Get the static size of this object.
    virtual std::size_t staticSize() const;

    //! Check invariants.
    bool checkInvariants() const;

    //! Print the sketch for debug.
    std::string print() const;

protected:
    using TBoolVec = std::vector<bool>;
    using TSizeVec = std::vector<std::size_t>;

protected:
    //! Reduce to the maximum permitted size.
    virtual void reduce();

    //! Get the target size for sketch post reduce.
    virtual std::size_t target() const;

    //! Reduce to the maximum permitted size.
    void reduce(CPRNG::CXorOShiro128Plus& rng,
                TFloatFloatPrVec& mergeCosts,
                TSizeVec& mergeCandidates,
                TBoolVec& stale);

    //! Sort and combine any co-located values.
    void orderAndDeduplicate();

    //! Compute the cost of combining \p vl and \p vr.
    static double cost(const TFloatFloatPr& vl, const TFloatFloatPr& vr);

    //! The maximum permitted size for the sketch.
    std::size_t maxSize() const { return m_MaxSize; }

private:
    //! Compute quantiles on the supplied knots.
    static void quantile(EInterpolation interpolation,
                         const TFloatFloatPrVec& knots,
                         double count,
                         double percentage,
                         double& result);

private:
    //! The style of interpolation to use.
    EInterpolation m_Interpolation;
    //! The maximum permitted size for the sketch.
    std::size_t m_MaxSize;
    //! The number of unsorted values.
    std::size_t m_Unsorted;
    //! The values and counts used as knot points to interpolate the c.d.f.
    TFloatFloatPrVec m_Knots;
    //! The total count of points in the sketch.
    double m_Count;
};

//! \brief Template wrapper for fixed size sketches which can be
//! default constructed.
template<CQuantileSketch::EInterpolation INTERPOLATION, std::size_t N>
class CFixedQuantileSketch final : public CQuantileSketch {
public:
    CFixedQuantileSketch() : CQuantileSketch(INTERPOLATION, N) {}

    //! Get a checksum of this object.
    //!
    //! \note Needs to be redeclared to work with CChecksum.
    std::uint64_t checksum(std::uint64_t seed = 0) const override {
        return this->CQuantileSketch::checksum(seed);
    }

    //! Debug the memory used by this object.
    //!
    //! \note Needs to be redeclared to work with CMemoryDebug.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override {
        this->CQuantileSketch::debugMemoryUsage(mem);
    }

    //! Get the memory used by this object.
    //!
    //! \note Needs to be redeclared to work with CMemory.
    std::size_t memoryUsage() const override {
        return this->CQuantileSketch::memoryUsage();
    }
};

//! \brief This tunes the quantile sketch for performance when space is less important.
//!
//! DESCRIPTION:\n
//! This uses around 2.5x the memory than `CQuantileSketch` but updating is around 2.0x
//! faster when using its default reduction factor.
class MATHS_EXPORT CFastQuantileSketch final : public CQuantileSketch {
public:
    CFastQuantileSketch(EInterpolation interpolation,
                        std::size_t size,
                        CPRNG::CXorOShiro128Plus rng = CPRNG::CXorOShiro128Plus{},
                        double reductionFraction = 0.8)
        : CQuantileSketch{interpolation, size}, m_Rng{rng}, m_ReductionFraction{reductionFraction} {
    }

    // We don't bother to checksum or persist and restore the bookkeeping state because
    // it is reinitialised at the start of each reduce.

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Get the static size of this object.
    std::size_t staticSize() const override;

private:
    using CQuantileSketch::reduce;

private:
    //! Reduce to the maximum permitted size.
    void reduce() override;

    //! Get the target size for sketch post reduce.
    std::size_t target() const override;

private:
    CPRNG::CXorOShiro128Plus m_Rng;
    TFloatFloatPrVec m_MergeCosts;
    TSizeVec m_MergeCandidates;
    TBoolVec m_Stale;
    double m_ReductionFraction;
};

//! Write to stream using print member.
inline std::ostream& operator<<(std::ostream& o, const CQuantileSketch& qs) {
    return o << qs.print();
}
}
}

#endif // INCLUDED_ml_maths_CQuantileSketch_h
