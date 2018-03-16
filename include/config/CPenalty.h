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

#ifndef INCLUDED_ml_config_CPenalty_h
#define INCLUDED_ml_config_CPenalty_h

#include <core/CoreTypes.h>

#include <config/ConfigTypes.h>
#include <config/ImportExport.h>

#include <boost/operators.hpp>
#include <boost/ref.hpp>
#include <boost/shared_ptr.hpp>

#include <cstddef>
#include <vector>

namespace ml {
namespace config {
class CAutoconfigurerParams;
class CCategoricalDataSummaryStatistics;
class CFieldStatistics;
class CDetectorSpecification;
class CDetectorRecord;
class CNumericDataSummaryStatistics;

//! \brief This specifies the interface for assigning a detector a penalty
//! in the range [0.0, 1.0].
//!
//! DESCRIPTION:\n
//! Rather than making hard decisions to keep or discard a detector, which
//! is effectively like having a discontinuous scoring function, auto-
//! configuration assigns each detector a score in the range [0, <max score>].
//! This is achieved by applying a product of smooth penalty functions on
//! the range [0,1], i.e. \f$f_p : D \rightarrow [0,1]\f$, where \f$D\f$ is
//! the set of all detectors for a given set of fields and the map is into,
//! to an initial max score.
//!
//! Note the set of all functions \f$\mathfrak{F} = \{f_p\}\f$ together
//! with the product operation that is \f$f_p * f_p'\f$, of which an example
//! would be \f$d \mapsto f_p(d)f_p'(d)\f$, is an abelian monoid. This means
//! we can get a partial ordering of all detectors by applying products of
//! penalties. It is also worth noting that the product is pointwise monotonic
//! decreasing in its "terms", i.e. for \f$f_p(d) * f_p'(d) \le f_p(d)\f$ for
//! any \f$f_p\f$, \f$f_p'\f$ and \f$d\f$, which implies dominance relations
//! between detectors which have a subset of the penalties which apply to
//! another detector and means as soon as its penalty is zero a detector can
//! be discarded. All the above discussion equally well applies if \f$f_p\f$
//! rounds down onto some finite set of points \f$k\f$ for
//! \f$k \in [\lfloor 1/\eps \rfloor]\f$.
//!
//! IMPLEMENTATION:\n
//! This uses the operator * to implement the product of two penalties
//! since is has the intuitively correct semantics. Note that the result
//! is a closure, i.e. (MyPenaltyA * MyPenaltyB) is a new CPenalty object
//! whose penalty function is the product of the penalty functions of
//! MyPenaltyA and MyPenaltyB.
class CONFIG_EXPORT CPenalty {
public:
    typedef std::vector<double> TDoubleVec;
    typedef std::vector<std::size_t> TSizeVec;
    typedef std::vector<core_t::TTime> TTimeVec;
    typedef std::vector<std::string> TStrVec;
    typedef boost::shared_ptr<CPenalty> TPenaltyPtr;
    typedef boost::shared_ptr<const CPenalty> TPenaltyCPtr;
    typedef std::vector<TPenaltyCPtr> TPenaltyCPtrVec;

    //! \brief Represents the result of multiplying penalties.
    class CClosure {
    public:
        CClosure(const CPenalty& penalty);

        //! Create a penalty on the heap from this closure.
        CPenalty* clone(void) const;

        //! Add a penalty to the closure.
        CClosure& add(const CPenalty& penalty);

        //! Get the closure's penalties.
        TPenaltyCPtrVec& penalties(void);

    private:
        //! The penalties in the closure.
        TPenaltyCPtrVec m_Penalties;
    };

public:
    CPenalty(const CAutoconfigurerParams& params);
    CPenalty(const CPenalty& other);
    explicit CPenalty(CClosure other);
    virtual ~CPenalty(void);

    //! Create a copy on the heap.
    virtual CPenalty* clone(void) const;

    //! Get the name of this penalty.
    virtual std::string name(void) const;

    //! Get the product penalty of this and \p rhs.
    const CPenalty& operator*=(const CPenalty& rhs);

    //! Get the product of this and the closure \p rhs.
    const CPenalty& operator*=(CClosure rhs);

    //! Compute the penalty to apply for the first property.
    void penalty(const CFieldStatistics& stats, double& penalty) const {
        std::string ignore;
        this->penalty(stats, penalty, ignore);
    }

    //! Compute the penalty to apply for the first property.
    void penalty(const CFieldStatistics& stats, double& penalty, std::string& description) const;

    //! Update the penalties of \p detector.
    void penalize(CDetectorSpecification& spec) const;

    //! Compute the score for \p penalty.
    static double score(double penalty);

    //! True if \p penalty forces the score to zero.
    static bool scoreIsZeroFor(double penalty);

protected:
    typedef boost::reference_wrapper<const CAutoconfigurerParams> TAutoconfigurerParamsCRef;

protected:
    //! Get the parameters.
    const CAutoconfigurerParams& params(void) const;

private:
    //! Not assignable.
    const CPenalty& operator=(const CPenalty& other);

    //! Compute the penalty based on a detector's field's statistics.
    //!
    //! \note No-op unless a derived class overrides it.
    virtual void penaltyFromMe(const CFieldStatistics& stats, double& penalty, std::string& description) const;

    //! Compute a penalty based a complete detector specification.
    //!
    //! \note No-op unless a derived class overrides it.
    virtual void penaltyFromMe(CDetectorSpecification& spec) const;

private:
    //! The parameters.
    TAutoconfigurerParamsCRef m_Params;

    //! The penalties.
    TPenaltyCPtrVec m_Penalties;
};

//! Multiply a two penalties.
CONFIG_EXPORT
CPenalty::CClosure operator*(const CPenalty& lhs, const CPenalty& rhs);
//! Multiply a closure by a penalty.
CONFIG_EXPORT
CPenalty::CClosure operator*(CPenalty::CClosure lhs, const CPenalty& rhs);
//! Multiply a penalty by a closure.
CONFIG_EXPORT
CPenalty::CClosure operator*(const CPenalty& lhs, CPenalty::CClosure rhs);
}
}

#endif // INCLUDED_ml_config_CPenalty_h
