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
#include <core/CStringCache.h>

#include <core/CLogger.h>

#include <string.h>


namespace ml
{
namespace core
{


const std::string CStringCache::EMPTY_STRING;


CStringCache::CStringCache(void)
    : m_HaveCopyOnWriteStrings(false)
{
    // Detect whether strings have copy-on-write semantics - if they don't then
    // this class may be of little value
    std::string test1("test copy-on-write");
    std::string test2(test1);
    if (test2.data() == test1.data())
    {
        m_HaveCopyOnWriteStrings = true;
    }
}

bool CStringCache::haveCopyOnWriteStrings(void) const
{
    return m_HaveCopyOnWriteStrings;
}

const std::string &CStringCache::stringFor(const char *str)
{
    // Stop processing NULL input immediately so that subsequent code doesn't
    // have to worry about NULL pointers
    if (str == 0)
    {
        return EMPTY_STRING;
    }

    return this->stringFor(str, ::strlen(str));
}

const std::string &CStringCache::stringFor(const char *str, size_t length)
{
    // Stop processing NULL input immediately so that subsequent code doesn't
    // have to worry about NULL pointers
    if (length == 0 || str == 0)
    {
        return EMPTY_STRING;
    }

    CCharPHash     hash(str, str + length);
    CCharPStrEqual equal(length);

    TStrUSetCItr iter = m_Cache.find(str, hash, equal);
    if (iter == m_Cache.end())
    {
        // This involves a temporary string creation, rehash, and other
        // processing.  This is why using this class is only worthwhile if a
        // small number of cached strings are seen repeatedly.
        iter = m_Cache.insert(std::string(str, length)).first;
    }

    return *iter;
}

size_t CStringCache::CStrHash::operator()(const std::string &str) const
{
    // It is essential that the result of this hash matches that of the method
    // below
    size_t hash(0);
    for (std::string::const_iterator iter = str.begin();
         iter != str.end();
         ++iter)
    {
        hash *= 17;
        hash += *iter;
    }
    return hash;
}

// Caller is responsible for ensuring that str is not NULL and end > str
inline
CStringCache::CCharPHash::CCharPHash(const char *str, const char *end)
    : m_Hash(0)
{
    // It is essential that the result of this hash matches that of the method
    // above
    size_t hash(*str);
    while (++str != end)
    {
        hash *= 17;
        hash += *str;
    }
    m_Hash = hash;
}

inline
size_t CStringCache::CCharPHash::operator()(const char *) const
{
    return m_Hash;
}

inline
CStringCache::CCharPStrEqual::CCharPStrEqual(size_t length)
    : m_Length(length)
{
}

// Caller is responsible for ensuring that lhs is not NULL
inline
bool CStringCache::CCharPStrEqual::operator()(const char *lhs, const std::string &rhs) const
{
    return m_Length == rhs.length() &&
           ::memcmp(lhs, rhs.data(), m_Length) == 0;
}


}
}

