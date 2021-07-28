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

#ifndef INCLUDED_ml_model_CEventData_h
#define INCLUDED_ml_model_CEventData_h

#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/optional.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace ml {
namespace model {

//! \brief The description of the event data corresponding to
//! a single record.
//!
//! DESCRIPTION:\n
//! Records are converted into event data for processing by the
//! model data gatherers. The event data comprises the person
//! which corresponds to either the unique by field, if no over
//! field is present, or the unique over field.
//!
//! If an over field is present, multiple by fields can be
//! aggregated and so extracted from each record. Therefore,
//! this supports collections of attribute identifiers. Also,
//! more than one metric field can be extracted from the same
//! record, so this supports a collection of metric values.
//!
//! Finally, each record has a unique time and person, although
//! with different semantics when an over field is and isn't
//! present, so this always holds the time and person identifier.
class MODEL_EXPORT CEventData {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TOptionalSize = boost::optional<std::size_t>;
    using TOptionalSizeVec = std::vector<TOptionalSize>;
    using TOptionalDouble = boost::optional<double>;
    // Fixed size array - one element per metric category
    using TDouble1VecArray = std::array<TDouble1Vec, model_t::NUM_METRIC_CATEGORIES>;
    // Second element in pair stores count
    using TDouble1VecArraySizePr = std::pair<TDouble1VecArray, std::size_t>;
    using TOptionalDouble1VecArraySizePr = boost::optional<TDouble1VecArraySizePr>;
    using TOptionalDouble1VecArraySizePrVec = std::vector<TOptionalDouble1VecArraySizePr>;
    using TOptionalStr = boost::optional<std::string>;
    using TOptionalStrVec = std::vector<TOptionalStr>;

public:
    //! Create uninitialized event data.
    CEventData();

    //! Efficiently swap the contents with \p other.
    void swap(CEventData& other);

    //! Reset to uninitialized state.
    void clear();

    //! Set the time.
    void time(core_t::TTime time);

    //! Set the person identifier.
    //!
    //! \warning There should only ever be one person identifier.
    bool person(std::size_t pid);

    //! Add an attribute identifier.
    void addAttribute(TOptionalSize cid = TOptionalSize());

    //! Add a function argument metric statistic value.
    void addValue(const TDouble1Vec& value = TDouble1Vec());

    //! Set the function argument string value.
    void stringValue(const std::string& value);

    //! Add an influencing field value.
    void addInfluence(const TOptionalStr& influence);

    //! Add a count only statistic.
    void addCountStatistic(size_t count);

    //! Add metric statistics.
    void addStatistics(const TDouble1VecArraySizePr& statistics);

    //! Get the event time.
    core_t::TTime time() const;

    //! Get the event person identifier.
    TOptionalSize personId() const;

    //! Get the unique attribute identifier.
    TOptionalSize attributeId() const;

    //! Get the function argument metric statistic value(s).
    const TDouble1VecArray& values() const;

    //! Get the function argument string value.
    const TOptionalStr& stringValue() const;

    //! Get the influencing field values.
    const TOptionalStrVec& influences() const;

    //! Sets the data to be explicit null
    void setExplicitNull();

    //! Is explicit null?
    bool isExplicitNull() const;

    //! Get the unique count of measurements comprising the statistic.
    TOptionalSize count() const;

    //! Get a description of the event data for debug.
    std::string print() const;

private:
    //! Read the \p i'th attribute identifier.
    TOptionalSize attributeId(std::size_t i) const;

    //! Read the \p i'th statistic value(s).
    const TDouble1VecArray& values(std::size_t i) const;

    //! Read the \p i'th statistic count.
    TOptionalSize count(std::size_t i) const;

private:
    //! The event time.
    core_t::TTime m_Time;
    //! The event person identifier.
    TOptionalSize m_Pid;
    //! The event attribute identifier(s).
    TOptionalSizeVec m_Cids;
    //! The event value(s).
    TOptionalDouble1VecArraySizePrVec m_Values;
    //! The function argument string value for this event.
    TOptionalStr m_StringValue;
    //! The influencing field values.
    TOptionalStrVec m_Influences;
    //! Is it an explicit null record?
    bool m_IsExplicitNull;
};
}
}

#endif // INCLUDED_ml_model_CEventData_h
