/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_maths_time_series_CDecompositionComponent_h
#define INCLUDED_ml_maths_time_series_CDecompositionComponent_h

#include <core/CMemoryUsage.h>
#include <core/CoreTypes.h>

#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/CPRNG.h>
#include <maths/common/CSpline.h>

#include <maths/time_series/ImportExport.h>

#include <array>
#include <cstddef>
#include <functional>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace time_series {

//! \brief Common functionality used by our decomposition component classes.
class MATHS_TIME_SERIES_EXPORT CDecompositionComponent {
public:
    using TVector2x1 = common::CVectorNx1<double, 2>;
    using TDoubleVec = std::vector<double>;
    using TFloatVec = std::vector<common::CFloatStorage>;
    using TSplineCRef = common::CSpline<std::reference_wrapper<const TFloatVec>,
                                        std::reference_wrapper<const TFloatVec>,
                                        std::reference_wrapper<const TDoubleVec>>;
    using TSplineRef =
        common::CSpline<std::reference_wrapper<TFloatVec>, std::reference_wrapper<TFloatVec>, std::reference_wrapper<TDoubleVec>>;

public:
    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Create by traversing a state document.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

protected:
    //! \brief A low memory representation of the value and variance splines.
    class MATHS_TIME_SERIES_EXPORT CPackedSplines {
    public:
        enum ESpline { E_Value = 0, E_Variance = 1 };

    public:
        using TTypeArray = std::array<common::CSplineTypes::EType, 2>;
        using TFloatVecArray = std::array<TFloatVec, 2>;
        using TDoubleVecArray = std::array<TDoubleVec, 2>;

    public:
        CPackedSplines(common::CSplineTypes::EType valueInterpolationType,
                       common::CSplineTypes::EType varianceInterpolationType);

        //! Create by traversing a state document.
        bool acceptRestoreTraverser(common::CSplineTypes::EBoundaryCondition boundary,
                                    core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! An efficient swap of the contents of two packed splines.
        void swap(CPackedSplines& other) noexcept;

        //! Check if the splines have been initialized.
        bool initialized() const;

        //! Clear the splines.
        void clear();

        //! Shift the spline values by \p shift.
        void shift(ESpline spline, double shift);

        //! Get a constant spline reference.
        TSplineCRef spline(ESpline spline) const;

        //! Get a writable spline reference.
        TSplineRef spline(ESpline spline);

        //! Get the splines' knot points.
        const TFloatVec& knots() const;

        //! Interpolate the value and variance functions on \p knots.
        void interpolate(const TDoubleVec& knots,
                         const TDoubleVec& values,
                         const TDoubleVec& variances,
                         common::CSplineTypes::EBoundaryCondition boundary);

        //! Get a checksum for this object.
        std::uint64_t checksum(std::uint64_t seed) const;

        //! Debug the memory used by the splines.
        void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

        //! Get the memory used by these splines.
        std::size_t memoryUsage() const;

    private:
        //! Check the state invariants after restoration
        //! Abort on failure.
        void checkRestoredInvariants() const;

    private:
        //! The splines' types.
        TTypeArray m_Types;
        //! The splines' knots.
        TFloatVec m_Knots;
        //! The splines' values.
        TFloatVecArray m_Values;
        //! The splines' curvatures.
        TDoubleVecArray m_Curvatures;
    };

protected:
    //! \param[in] maxSize The maximum number of component buckets.
    //! \param[in] boundaryCondition The boundary condition to use for the splines.
    //! \param[in] valueInterpolationType The style of interpolation to use for
    //! computing values.
    //! \param[in] varianceInterpolationType The style of interpolation to use for
    //! computing variances.
    CDecompositionComponent(std::size_t maxSize,
                            common::CSplineTypes::EBoundaryCondition boundaryCondition,
                            common::CSplineTypes::EType valueInterpolationType,
                            common::CSplineTypes::EType varianceInterpolationType);

    //! An efficient swap of the contents of two components.
    void swap(CDecompositionComponent& other);

    //! Check if the seasonal component has been estimated.
    bool initialized() const;

    //! Clear all data.
    void clear();

    //! Update the interpolation of the bucket values.
    //!
    //! \param[in] knots The spline knot points.
    //! \param[in] values The values at the spline knot points.
    //! \param[in] variances The variances at the spline knot points.
    void interpolate(const TDoubleVec& knots, const TDoubleVec& values, const TDoubleVec& variances);

    //! Shift the component's values by \p shift.
    void shiftLevel(double shift);

    //! Interpolate the function at \p time.
    //!
    //! \param[in] offset The offset for which to get the value.
    //! \param[in] n The bucket count containing \p offset.
    //! \param[in] confidence The symmetric confidence interval for the variance
    //! as a percentage.
    TVector2x1 value(double offset, double n, double confidence) const;

    //! Get the mean value of the function.
    double meanValue() const;

    //! Get the variance of the residual about the function at \p time.
    //!
    //! \param[in] offset The offset for which to get the variance.
    //! \param[in] n The bucket count containing \p offset.
    //! \param[in] confidence The symmetric confidence interval for the
    //! variance as a percentage.
    TVector2x1 variance(double offset, double n, double confidence) const;

    //! Get the mean variance of the function residuals.
    double meanVariance() const;

    //! Get the maximum size to use for the bucketing.
    std::size_t maxSize() const;

    //! Get the boundary condition to use when interpolating.
    common::CSplineTypes::EBoundaryCondition boundaryCondition() const;

    //! Get the value spline.
    TSplineCRef valueSpline() const;

    //! Get the variance spline.
    TSplineCRef varianceSpline() const;

    //! Get the underlying splines representation.
    const CPackedSplines& splines() const;

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed) const;

private:
    //! The minimum permitted size for the points sketch.
    static const std::size_t MIN_MAX_SIZE;

private:
    //! The maximum number of buckets to use to cover the period.
    std::size_t m_MaxSize;

    //! The boundary condition to use for the splines.
    common::CSplineTypes::EBoundaryCondition m_BoundaryCondition;

    //! The spline we fit through the function points and the function point
    //! residual variances.
    CPackedSplines m_Splines;

    //! The mean value in the period.
    double m_MeanValue;

    //! The mean residual variance in the period.
    double m_MeanVariance;
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CDecompositionComponent_h
