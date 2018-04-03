/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_core_CStringCache_h
#define INCLUDED_ml_core_CStringCache_h

#include <core/ImportExport.h>

#include <boost/unordered_set.hpp>

#include <string>


namespace ml
{
namespace core
{


//! \brief
//! A cache of strings that can be reused
//!
//! DESCRIPTION:\n
//! Rather than repeatedly construct strings from char
//! pointers, there are occasions where it may be preferable
//! to reuse an existing string object with the same contents
//! as the char pointer is pointing to.  Looking up an
//! existing string and returning a reference to it avoids
//! calls to malloc() and free(), which are expensive and
//! cause contention between threads.
//!
//! This approach is very beneficial on platforms where the
//! STL string class uses a copy-on-write implementation.  On
//! other platforms it will probably WORSEN performance, so
//! ideally the calling code should check at runtime what the
//! situation is and bypass this class for platforms that
//! don't have copy-on-write strings.
//!
//! This class should only be used where a small number of
//! strings are expected to be seen repeatedly.  If many
//! strings are different, this class will be a liability
//! rather than a benefit.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Using a hash set for the cache of strings.
//!
//! It is absolutely crucial that no temporary strings are
//! created when looking up a string that is already in the
//! cache - doing so would completely defeat the purpose of
//! this class!
//!
//! This class is not thread safe - it is up to the caller
//! to ensure each instance of this class is only used from
//! within a single thread.
//!
//! Boost's hash function applied to an empty string returns
//! a non-zero hash, which would be hard to reproduce for a
//! range of characters, hence we use our own hash algorithm.
//!
class CORE_EXPORT CStringCache
{
    public:
        //! Constructor detects whether copy-on-write strings are in use
        CStringCache(void);

        //! Does the current platform use copy-on-write strings?  If it
        //! doesn't, it's probably best not to use any further functionality
        //! of this class.
        bool haveCopyOnWriteStrings(void) const;

        //! Look up a char pointer when the length is not known
        const std::string &stringFor(const char *str);

        //! If the length is already known the hash calculation can be more
        //! efficient
        const std::string &stringFor(const char *str, size_t length);

    private:
        //! String hash that uses the same formula as CCharPHash below.
        //! Boost's hash function applied to an empty string returns a non-zero
        //! hash, which would be hard to reproduce for a range of characters,
        //! hence using a hand coded hash functor.
        class CStrHash : public std::unary_function<std::string, size_t>
        {
            public:
                size_t operator()(const std::string &str) const;
        };

        //! Class to hash a range of characters on construction to save
        //! calculating the length in operator().  Does NOT construct a
        //! temporary string object to create the hash.
        class CCharPHash : public std::unary_function<const char *, size_t>
        {
            public:
                //! Store the given hash
                CCharPHash(const char *str, const char *end);

                //! Return the hash computed in the constructor regardless of
                //! what argument is passed.
                size_t operator()(const char *) const;

            private:
                size_t m_Hash;
        };

        //! Check for equality between a char pointer and a string without
        //! constructing a temporary string
        class CCharPStrEqual : public std::binary_function<const char *, std::string, bool>
        {
            public:
                //! Cache the char pointer length to speed comparisons
                CCharPStrEqual(size_t length);

                bool operator()(const char *lhs, const std::string &rhs) const;

            private:
                size_t m_Length;
        };

    private:
        //! Flag to record whether the current platform has copy-on-write
        //! strings
        bool m_HaveCopyOnWriteStrings;

        using TStrUSet = boost::unordered_set<std::string, CStrHash>;
        using TStrUSetCItr = TStrUSet::const_iterator;

        //! The cache of strings
        TStrUSet m_Cache;

        //! String to return when passed a NULL pointer
        static const std::string EMPTY_STRING;
};


}
}

#endif // INCLUDED_ml_core_CStringCache_h

