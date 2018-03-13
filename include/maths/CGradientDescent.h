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

#ifndef INCLUDED_ml_maths_CGradientDescent_h
#define INCLUDED_ml_maths_CGradientDescent_h

#include <core/CLogger.h>
#include <core/CNonCopyable.h>

#include <maths/CLinearAlgebra.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <vector>

namespace ml {
namespace maths {

//! \brief Implements gradient descent with momentum.
//!
//! DESCRIPTION\n
//! \see https://en.wikipedia.org/wiki/Gradient_descent.
class MATHS_EXPORT CGradientDescent {
public:
    typedef std::vector<double> TDoubleVec;
    typedef CVector<double> TVector;

    //! \brief The interface for the function calculation.
    class MATHS_EXPORT CFunction {
    public:
        virtual ~CFunction(void);
        virtual bool operator()(const TVector &x, double &result) const = 0;
    };

    //! \brief The interface for the gradient calculation.
    class MATHS_EXPORT CGradient {
    public:
        virtual ~CGradient(void);
        virtual bool operator()(const TVector &x, TVector &result) const = 0;
    };

    //! \brief Computes the gradient using the central difference
    //! method.
    //!
    //! DESCRIPTION:\n
    //! \see https://en.wikipedia.org/wiki/Finite_difference.
    class MATHS_EXPORT CEmpiricalCentralGradient : public CGradient, private core::CNonCopyable {
    public:
        CEmpiricalCentralGradient(const CFunction &f, double eps);

        virtual bool operator()(const TVector &x, TVector &result) const;

    private:
        //! The shift used to get the offset points.
        double m_Eps;
        //! The function for which to compute the gradient.
        const CFunction &m_F;
        //! A placeholder for the shifted points.
        mutable TVector xShiftEps;
    };

public:
    CGradientDescent(double learnRate, double momentum);

    //! Set the learn rate.
    void learnRate(double learnRate);

    //! Set the momentum.
    void momentum(double momentum);

    //! Run gradient descent for \p n steps.
    //!
    //! \param[in] n The number of steps to use.
    //! \param[in] x0 The starting point for the argument of the function
    //! to minimize.
    //! \param[in] f The function to minimize.
    //! \param[in] gf The gradient oracle of the function to minimize.
    //! \param[out] xBest Filled in with the minimum function value argument
    //! visited.
    //! \param[out] fi Filled in with the sequence of function values.
    bool run(std::size_t n,
             const TVector &x0,
             const CFunction &f,
             const CGradient &gf,
             TVector &xBest,
             TDoubleVec &fi);

private:
    //! The multiplier of the unit vector along the gradient.
    double m_LearnRate;

    //! The proportion of the previous step to add.
    double m_Momentum;

    //! The last step.
    TVector m_PreviousStep;
};
}
}

#endif// INCLUDED_ml_maths_CGradientDescent_h
