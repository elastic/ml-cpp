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
#include "CTopLevelDomainDb.h"

#include <core/CLogger.h>
#include <core/CTextFileWatcher.h>

#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>

namespace ml {
namespace domain_name_entropy {

const std::string CTopLevelDomainDb::PUNY_CODE = "xn--";
const std::string CTopLevelDomainDb::PERIOD = ".";

CTopLevelDomainDb::CTopLevelDomainDb(const std::string& effectiveTldNamesFileName)
    : m_EffectiveTldNamesFileName(effectiveTldNamesFileName) {
}

bool CTopLevelDomainDb::init(void) {
    core::CTextFileWatcher watcher;

    if (watcher.init(m_EffectiveTldNamesFileName, "\r?\n", core::CTextFileWatcher::E_Start) ==
        false) {
        LOG_ERROR("Can not open " << m_EffectiveTldNamesFileName);
        return false;
    }

    std::string remainder;
    if (watcher.readAllLines(boost::bind(&CTopLevelDomainDb::readLine, this, _1), remainder) ==
        false) {
        LOG_ERROR("Error reading " << m_EffectiveTldNamesFileName);
        return false;
    }

    // deal with remainder
    return this->readLine(remainder);
}

bool CTopLevelDomainDb::readLine(const std::string& line) {
    /*
    https://publicsuffix.org/list/#list-format

    Definitions
    The Public Suffix List consists of a series of lines, separated by \n.
    Each line is only read up to the first whitespace; entire lines can also be commented using //.
    Each line which is not entirely whitespace or begins with a comment contains a rule.
    A rule may begin with a "!" (exclamation mark). If it does, it is labelled as a "exception rule"
    and then treated as if the exclamation mark is not present. A domain or rule can be split into a
    list of labels using the separator "." (dot). The separator is not part of any of the labels. A
    domain is said to match a rule if, when the domain and rule are both split, and one compares the
    labels from the rule to the labels from the domain, beginning at the right hand end, one finds
    that for every pair either they are identical, or that the label from the rule is "*" (star).
    The domain may legitimately have labels remaining at the end of this matching process.
    */
    std::string trimmedLine = boost::algorithm::trim_copy(line);

    if (trimmedLine.empty()) {
        return true;
    }

    // Is it a comment?
    if (trimmedLine.substr(0, 2) == "//") {
        return true;
    }

    // Put in lower_case for consistency
    boost::algorithm::to_lower(trimmedLine);

    // Is it an exception?
    if (trimmedLine.front() == '!') {
        if (trimmedLine.size() < 2) {
            LOG_WARN(trimmedLine << " not valid - ignoring.");
            return true;
        }

        // Hash without first '!' character
        if (m_EffectiveTldNamesExceptions.insert(trimmedLine.substr(1, std::string::npos)).second ==
            false) {
            // Benign warning
            LOG_WARN("Inconsistency in " << m_EffectiveTldNamesFileName << " duplicate "
                                         << trimmedLine);
        }
        return true;
    }

    // Is it a wildcard?
    if (trimmedLine.front() == '*') {
        if (trimmedLine.size() < 2) {
            LOG_WARN(trimmedLine << " not valid - ignoring.");
            return true;
        }

        // Hash without first '*.' character
        if (m_EffectiveTldNamesWildcards.insert(trimmedLine.substr(2, std::string::npos)).second ==
            false) {
            // Benign warning
            LOG_WARN("Inconsistency in " << m_EffectiveTldNamesFileName << " duplicate "
                                         << trimmedLine);
        }
        return true;
    }

    // Normal insert
    if (m_EffectiveTldNames.insert(trimmedLine).second == false) {
        // Benign warning
        LOG_WARN("Inconsistency in " << m_EffectiveTldNamesFileName << " duplicate "
                                     << trimmedLine);
    }

    return true;
}

bool CTopLevelDomainDb::registeredDomainName(const std::string& host,
                                             std::string& registeredHostName) const {
    std::string subDomain;
    std::string domain;
    std::string suffix;

    this->splitHostName(host, subDomain, domain, suffix);
    // LOG_DEBUG(host << " sub:" << subDomain << " dom:" << domain << " suf:" << suffix);

    // Be strict here to comply with rules.
    // The caller can choose to use the host name instead of the domain
    // name if need be.
    if (!suffix.empty() && !domain.empty()) {
        registeredHostName = domain + PERIOD + suffix;
        return true;
    }

    return false;
}

void CTopLevelDomainDb::splitHostName(const std::string& host,
                                      std::string& subDomain,
                                      std::string& domain,
                                      std::string& suffix) const {
    if (m_EffectiveTldNames.empty()) {
        LOG_ERROR("No rules. Call ::init to initialize object.");
    }

    bool isPunyCode = false;

    if (host.find(PUNY_CODE) != std::string::npos) {
        isPunyCode = true;

        // Translate puny_code to idna as lookup file is in idna
        LOG_FATAL("NOT HANDLED YET");
    }

    // First do all of this in lower cass
    std::string lowerHost = boost::algorithm::to_lower_copy(host);

    this->extract(lowerHost, subDomain, domain, suffix);

    // Trasnlate back to ascii
    if (isPunyCode) {
        LOG_FATAL("NOT HANDLED YET");
    }
}

void CTopLevelDomainDb::extract(const std::string& str,
                                std::string& subDomain,
                                std::string& domain,
                                std::string& suffix) const {
    /*
    https://publicsuffix.org/list/#list-format

    Algorithm
    Match domain against all rules and take note of the matching ones.
    If no rules match, the prevailing rule is "*".
    If more than one rule matches, the prevailing rule is the one which is an exception rule.
    If there is no matching exception rule, the prevailing rule is the one with the most labels.
    If the prevailing rule is a exception rule, modify it by removing the leftmost label.
    The public suffix is the set of labels from the domain which directly match the labels of the
    prevailing rule (joined by dots). The registered or registrable domain is the public suffix plus
    one additional label.

    Some examples:


    */
    TSizeTypeVec periods;

    std::string::size_type pos(0);

    for (;;) {
        std::string maybeSuffix = str.substr(pos, std::string::npos);

        switch (this->isSuffixTld(maybeSuffix)) {
        case E_Rule: {
            CTopLevelDomainDb::ruleDomains(str, periods, subDomain, domain, suffix);
            return;
        }
        case E_WildcardRule: {
            CTopLevelDomainDb::wildcardDomains(str, periods, subDomain, domain, suffix);
            return;
        }
        case E_ExceptionRule: {
            // Force an extra interation
            pos = str.find(PERIOD, pos);
            if (pos == std::string::npos) {
                LOG_WARN("Unexpected domain for exception rule: '" << str);
            }

            pos = pos + PERIOD.size();

            periods.push_back(pos);

            CTopLevelDomainDb::exceptionDomains(str, periods, subDomain, domain, suffix);
            return;
        }
        case E_NoMatch: {
            // fall through
            break;
        }
        }

        pos = str.find(PERIOD, pos);
        if (pos == std::string::npos) {
            break;
        }

        pos = pos + PERIOD.size();

        periods.push_back(pos);
    }

    // If no match all the way, match "*"
    CTopLevelDomainDb::ruleDomains(str, periods, subDomain, domain, suffix);
}

// Some helpers
namespace {

typedef std::vector<std::string::size_type> TSizeTypeVec;

std::string::size_type _last(const TSizeTypeVec& v) {
    if (v.empty()) {
        return 0;
    }

    return v[v.size() - 1];
}

std::string::size_type _penultimate(const TSizeTypeVec& v) {
    if (v.size() < 2) {
        return 0;
    }

    return v[v.size() - 2];
}

std::string::size_type _antepenultimate(const TSizeTypeVec& v) {
    if (v.size() < 3) {
        return 0;
    }

    return v[v.size() - 3];
}
}

void CTopLevelDomainDb::ruleDomains(const std::string& str,
                                    const TSizeTypeVec& periods,
                                    std::string& subDomain,
                                    std::string& domain,
                                    std::string& suffix)

{
    std::string::size_type last = _last(periods);
    std::string::size_type penultimate = _penultimate(periods);

    if (last > 0) {
        if (penultimate > 0) {
            subDomain = str.substr(0, penultimate - PERIOD.size());
        }
        domain = str.substr(penultimate, last - penultimate - PERIOD.size());
    }
    suffix = str.substr(last, std::string::npos);
}

void CTopLevelDomainDb::wildcardDomains(const std::string& str,
                                        const TSizeTypeVec& periods,
                                        std::string& subDomain,
                                        std::string& domain,
                                        std::string& suffix)

{
    std::string::size_type last = _penultimate(periods);
    std::string::size_type penultimate = _antepenultimate(periods);

    if (last > 0) {
        if (penultimate > 0) {
            subDomain = str.substr(0, penultimate - PERIOD.size());
        }
        domain = str.substr(penultimate, last - penultimate - PERIOD.size());
    }
    suffix = str.substr(last, std::string::npos);
}

void CTopLevelDomainDb::exceptionDomains(const std::string& str,
                                         const TSizeTypeVec& periods,
                                         std::string& subDomain,
                                         std::string& domain,
                                         std::string& suffix) {
    CTopLevelDomainDb::ruleDomains(str, periods, subDomain, domain, suffix);
}

CTopLevelDomainDb::ERuleType CTopLevelDomainDb::isSuffixTld(const std::string& suffix) const {
    // If more than one rule matches, the prevailing rule is the one which is an exception rule.
    // - check exception rules first
    // If the prevailing rule is a exception rule, modify it by removing the leftmost label.
    if (m_EffectiveTldNamesExceptions.find(suffix) != m_EffectiveTldNamesExceptions.end()) {
        return E_ExceptionRule;
    }
    // If there is no matching exception rule, the prevailing rule is the one with the most labels.
    if (m_EffectiveTldNames.find(suffix) != m_EffectiveTldNames.end()) {
        return E_Rule;
    }
    // If there is no matching exception rule, the prevailing rule is the one with the most labels.
    if (m_EffectiveTldNamesWildcards.find(suffix) != m_EffectiveTldNamesWildcards.end()) {
        return E_WildcardRule;
    }
    // If no rules match, the prevailing rule is "*".
    return E_NoMatch;
}
}
}
