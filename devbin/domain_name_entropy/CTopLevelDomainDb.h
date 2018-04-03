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
#ifndef INCLUDED_ml_domain_name_entropy_CTopLevelDomainDb_h
#define INCLUDED_ml_domain_name_entropy_CTopLevelDomainDb_h

#include <set>
#include <string>
#include <vector>

#include <core/CNonCopyable.h>

namespace ml {
namespace domain_name_entropy {

//! \brief
//! Split a domain into suffix, registered domain and subdomain
//!
//! DESCRIPTION:\n
//! Split a domain into suffix, registered domain and subdomain
//!
//! e.g.
//! forums.news.cnn.com
//! (subdomain='forums.news', domain='cnn', suffix='com')
//! forums.bbc.co.uk
//! (subdomain='forums', domain='bbc', suffix='co.uk')
//! www.worldbank.org.kg
//! (subdomain='www', domain='worldbank', suffix='org.kg')
//!
//! "There was and remains no algorithmic method of finding the
//! highest level at which a domain may be registered for a
//! particular top-level domain (the policies differ with each registry),
//! the only method is to create a list. This is the aim of the Public
//! Suffix List." (https://publicsuffix.org/learn/).
//!
//! IMPLEMENTATION DECISIONS:\n
//! Reference is 'https://publicsuffix.org/'
//!
//! Reads a static file, but could also get file from:
//!
//! https://publicsuffix.org/list/effective_tld_names.dat
//! or
//! https://raw.github.com/mozilla/gecko-dev/master/netwerk/dns/effective_tld_names.dat
//!
//! The matching algorithm is defined in:
//!
//! https://publicsuffix.org/list/#list-format
//!
//! Currently, all private domains are included (e.g. blogspot.com)
//! These can be excluded if required.
//!
//! Most open source implmentations of don't work effectively and
//! don't pass the tests from:
//!
//! http://mxr.mozilla.org/mozilla-central/source/netwerk/test/unit/data/test_psl.txt?raw=1
//!
//! A python implementation of this is:
//! https://github.com/john-kurkowski/tldextract
//!
//! + additional php implementation is here:
//!
//! http://alandix.com/code/public-suffix/check/check.php
//!
//! + this go implementation seems the most complete:
//!
//! https://code.google.com/p/go/source/browse/publicsuffix/list_test.go?repo=net
//!
class CTopLevelDomainDb : private core::CNonCopyable {
public:
    CTopLevelDomainDb(const std::string& effectiveTldNamesFileName);

    //! Create DB
    bool init(void);

    //! get the 'registered' domain name
    //! note: this may return an empty string if the host
    //! name is not a valid domain name.
    //! for example,
    //! - if the host name is blank
    //! - if the host name is a single label: "testmachine"
    //! - if the host name is a suffix: "co.uk", "s3.amazonaws.com"
    //! - if the host name is a wildcard: "test.ck" (there is a rule *.ck) so test.ck is a public suffix
    bool registeredDomainName(const std::string& host, std::string& registereddomainname) const;

    //! Split a host name into
    //! 'subdomain' - token above domain
    //! 'domain' - next token above suffix (domain+suffix) is registered domain
    //! 'suffix' - TLD suffix (if available)
    void splitHostName(const std::string& host, std::string& subDomain, std::string& domain, std::string& suffix) const;

private:
    enum ERuleType { E_ExceptionRule = 0, E_Rule, E_WildcardRule, E_NoMatch };

    ERuleType isSuffixTld(const std::string& suffix) const;

    //! Read a line from the tld file
    bool readLine(const std::string&);

    //! Internal extract domains using rules
    void extract(const std::string& str, std::string& subDomain, std::string& domain, std::string& suffix) const;

    typedef std::vector<std::string::size_type> TSizeTypeVec;

    //! If a normal rule matches, split domain
    static void
    ruleDomains(const std::string& str, const TSizeTypeVec& periods, std::string& subDomain, std::string& domain, std::string& suffix);

    //! If a wildcard rule matches, split domain
    static void
    wildcardDomains(const std::string& str, const TSizeTypeVec& periods, std::string& subDomain, std::string& domain, std::string& suffix);

    //! If an exception rule matches, split domain
    static void
    exceptionDomains(const std::string& str, const TSizeTypeVec& periods, std::string& subDomain, std::string& domain, std::string& suffix);

private:
    static const std::string PERIOD;
    static const std::string PUNY_CODE;

private:
    const std::string m_EffectiveTldNamesFileName;

    typedef std::set<std::string> TStrSet;

    TStrSet m_EffectiveTldNames;
    TStrSet m_EffectiveTldNamesExceptions;
    TStrSet m_EffectiveTldNamesWildcards;
};
}
}

#endif // INCLUDED_ml_domain_name_entropy_CTopLevelDomainDb_h
