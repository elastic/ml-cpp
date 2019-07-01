/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStatePersistInserter_h
#define INCLUDED_ml_core_CStatePersistInserter_h

#include <core/CNonCopyable.h>
#include <core/CStringUtils.h>
#include <core/ImportExport.h>

#include <limits>
#include <string>

namespace ml {
namespace core {

//! \brief
//! Convenience container class for holding persistence tags.
//!
//! DESCRIPTION:\n
//! The standard format for persistence tags is short, a single letter,
//! keeping the persistence dumps as small as possible. Presenting
//! some or all of the persistence state in a human-readable form
//! requires the use of longer, meaningful tags corresponding to the
//! short forms.
//!
//! This class is a simple wrapper around the short and long form of associated tags.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Default copy & move construction and assignment.
//! Conversion to/from std::string.
//! std::string comparison operators.
//!
//!
//! Where there is any ambiguity over which tag to choose the short form is favoured.
//! This is since currently the long form of the tag names are not required to be restored
//! from state - only persisted.
//!
class CORE_EXPORT CPersistenceTag {
public:
    //! Construct from a single tag - set both short and long forms to the same value.
    explicit CPersistenceTag(const std::string& tag)
        : m_ShortTag(tag), m_LongTag(tag) {}

    //! Construct from a pair of tags in short and long form.
    CPersistenceTag(const std::string shortTag, const std::string& longTag)
        : m_ShortTag(shortTag), m_LongTag(longTag) {}

    //! Compiler generated defaults
    CPersistenceTag() = default;
    ~CPersistenceTag() = default;

    CPersistenceTag& operator=(const CPersistenceTag& other) = default;
    CPersistenceTag& operator=(CPersistenceTag&& other) = default;

    CPersistenceTag(const CPersistenceTag& other) = default;
    CPersistenceTag(CPersistenceTag&& other) = default;

    //! Get the desired form of the tag.
    //!
    //! \param[in] readableTags Whether the short or long (readable) form of the tag is desired.
    std::string name(bool readableTags) const {
        return readableTags ? m_LongTag : m_ShortTag;
    }

    //! Conversion operator returning the short tag form.
    operator std::string() const { return m_ShortTag; }

    //! Subscript operator for the short tag form.
    auto operator[](std::size_t i) const { return m_ShortTag[i]; }

    //! Comparison operator for a single tag. Return true if either short or long tags match.
    bool operator==(const std::string& rhs) const {
        return rhs == m_ShortTag || rhs == m_LongTag;
    }

    //! Check if two tags are identical.
    bool operator==(const CPersistenceTag& rhs) const {
        return rhs.m_ShortTag == m_ShortTag && rhs.m_LongTag == m_LongTag;
    }

private:
    std::string m_ShortTag;
    std::string m_LongTag;

private:
    friend CORE_EXPORT bool operator==(const std::string& lhs, const CPersistenceTag& rhs);
    friend CORE_EXPORT bool operator!=(const std::string& lhs, const CPersistenceTag& rhs);
};
using TPersistenceTag = CPersistenceTag;

CORE_EXPORT bool operator==(const std::string& lhs, const TPersistenceTag& rhs);
CORE_EXPORT bool operator!=(const std::string& lhs, const TPersistenceTag& rhs);

//! \brief
//! Abstract interface for persisting state.
//!
//! DESCRIPTION:\n
//! Classes that need to persist state may accept this interface
//! as a means to generically state which values they need to
//! persist.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Not copyable.
//!
//! All values are stored as strings.
//!
class CORE_EXPORT CStatePersistInserter : private CNonCopyable {
public:
    //! Virtual destructor for abstract class
    virtual ~CStatePersistInserter();

    //! Store a name/value
    virtual void insertValue(const std::string& name, const std::string& value) = 0;

    //! Store a name/value with choice of tag format
    virtual void insertValue(const TPersistenceTag& tag, const std::string& value) {
        this->insertValue(tag.name(this->readableTags()), value);
    }

    //! Store an arbitrary type that can be converted to a string
    template<typename TYPE>
    void insertValue(const std::string& name, const TYPE& value) {
        this->insertValue(name, CStringUtils::typeToString(value));
    }

    //! Store an arbitrary type that can be converted to a string
    //! with choice of tag format
    template<typename TYPE>
    void insertValue(const TPersistenceTag& name, const TYPE& value) {
        this->insertValue(name.name(this->readableTags()), value);
    }

    //! Store a floating point number with a given level of precision
    void insertValue(const std::string& name, double value, CIEEE754::EPrecision precision);

    //! Store a floating point number with a given level of precision
    //! with choice of tag format
    void insertValue(const TPersistenceTag& name, double value, CIEEE754::EPrecision precision) {
        this->insertValue(name.name(this->readableTags()), value, precision);
    }

    //! Store a nested level of state, to be populated by the supplied
    //! function or function object
    template<typename FUNC>
    void insertLevel(const std::string& name, FUNC f) {
        CAutoLevel level(name, *this);
        f(*this);
    }

    //! Store a nested level of state, to be populated by the supplied
    //! function or function object - with choice of tag format
    template<typename FUNC>
    void insertLevel(const TPersistenceTag& name, FUNC f) {
        this->insertLevel(name.name(this->readableTags()), f);
    }

    //! Get whether short or long (readable) tag names are desired for persistence.
    virtual bool readableTags() const { return false; }

protected:
    //! Start a new level with the given name
    virtual void newLevel(const std::string& name) = 0;

    //! End the current level
    virtual void endLevel() = 0;

private:
    //! Class to implement RAII for moving to the next level
    class CORE_EXPORT CAutoLevel : private CNonCopyable {
    public:
        CAutoLevel(const std::string& name, CStatePersistInserter& inserter);
        ~CAutoLevel();

    private:
        CStatePersistInserter& m_Inserter;
    };
};
}
}

#endif // INCLUDED_ml_core_CStatePersistInserter_h
