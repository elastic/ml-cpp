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
#include "CTopLevelDomainDbTest.h"

#include <core/CLogger.h>

#include "../CTopLevelDomainDb.h"

using namespace ml;
using namespace domain_name_entropy;

CppUnit::Test *CTopLevelDomainDbTest::suite()
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CTopLevelDomainDbTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CTopLevelDomainDbTest>(
                                   "CTopLevelDomainDbTest::testSimpleTestCases",
                                   &CTopLevelDomainDbTest::testSimpleTestCases) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTopLevelDomainDbTest>(
                                   "CTopLevelDomainDbTest::testMozillaTestCases",
                                   &CTopLevelDomainDbTest::testMozillaTestCases) );
    return suiteOfTests;
}

namespace
{
void checkPublicSuffix(const std::string &fullName, 
                       const std::string &registeredNameExpected,
                       CTopLevelDomainDb &tldDb)
{
    std::string registeredName;

    tldDb.registeredDomainName(fullName, registeredName);

    CPPUNIT_ASSERT_EQUAL(registeredNameExpected, registeredName);
}

void testDomainSplit(const std::string &subDomainExpected,
                     const std::string &domainExpected,
                     const std::string &suffixExpected,
                     const std::string &hostName, 
                     const CTopLevelDomainDb &tldDb)
{
    std::string subDomain;
    std::string domain;
    std::string suffix;

    tldDb.splitHostName(hostName, subDomain, domain, suffix);

    LOG_DEBUG(hostName << ":" << subDomain << ":" << domain << ":" << suffix);

    CPPUNIT_ASSERT_EQUAL(subDomainExpected, subDomain);
    CPPUNIT_ASSERT_EQUAL(domainExpected, domain);
    CPPUNIT_ASSERT_EQUAL(suffixExpected, suffix);
}

}

void CTopLevelDomainDbTest::testSimpleTestCases(void)
{
    CTopLevelDomainDb tldDb("../effective_tld_names.txt");

    CPPUNIT_ASSERT(tldDb.init());

    // Test cases from https://github.com/john-kurkowski/tldextract/tree/master/tldextract/tests
    testDomainSplit("www", "google", "com", "www.google.com", tldDb);
    testDomainSplit("www", "theregister", "co.uk", "www.theregister.co.uk", tldDb);
    testDomainSplit("", "gmail", "com", "gmail.com", tldDb);
    testDomainSplit("media.forums", "theregister", "co.uk","media.forums.theregister.co.uk", tldDb);
    testDomainSplit("www", "www", "com", "www.www.com", tldDb);
    testDomainSplit("", "www", "com", "www.com", tldDb);
    testDomainSplit("", "", "internalunlikelyhostname", "internalunlikelyhostname", tldDb);
    testDomainSplit("", "internalunlikelyhostname", "bizarre", "internalunlikelyhostname.bizarre", tldDb);
    testDomainSplit("", "internalunlikelyhostname", "info", "internalunlikelyhostname.info", tldDb);
    testDomainSplit("", "internalunlikelyhostname", "information", "internalunlikelyhostname.information", tldDb);
    //testDomainSplit("", "216.22.0.192", "", "216.22.0.192", tldDb);
    testDomainSplit("216.22", "project", "coop", "216.22.project.coop", tldDb);
    testDomainSplit("", "", "1\xe9", "1\xe9", tldDb);
    //testDomainSplit("", "россия", "рф", "xn--h1alffa9f.xn--p1ai", tldDb);
    testDomainSplit("", "", "", "", tldDb);
    testDomainSplit("www", "parliament", "uk", "www.parliament.uk", tldDb);
    testDomainSplit("www", "parliament", "co.uk", "www.parliament.co.uk", tldDb);
    testDomainSplit("www", "cgs", "act.edu.au", "www.cgs.act.edu.au", tldDb);
    testDomainSplit("www", "google", "com.au", "www.google.com.au", tldDb);
    testDomainSplit("www", "metp", "net.cn", "www.metp.net.cn", tldDb);
    //testDomainSplit("www", "example", "com", "www.example.com.", tldDb);
    //testDomainSplit("waiterrant", "blogspot", "com", "waiterrant.blogspot.com", tldDb);
}


void CTopLevelDomainDbTest::testMozillaTestCases(void)
{
    CTopLevelDomainDb tldDb("../effective_tld_names.txt");

    CPPUNIT_ASSERT(tldDb.init());

// Any copyright is dedicated to the Public Domain.
// http://creativecommons.org/publicdomain/zero/1.0/

// std::string() input.
checkPublicSuffix(std::string(), std::string(), tldDb);
// Mixed case.
checkPublicSuffix("COM", std::string(), tldDb);
checkPublicSuffix("example.COM", "example.com", tldDb);
checkPublicSuffix("WwW.example.COM", "example.com", tldDb);
// Leading dot.
//checkPublicSuffix(".com", std::string(), tldDb);
//checkPublicSuffix(".example", std::string(), tldDb);
//checkPublicSuffix(".example.com", std::string(), tldDb);
//checkPublicSuffix(".example.example", std::string(), tldDb);
// Unlisted TLD.
checkPublicSuffix("example", std::string(), tldDb);
checkPublicSuffix("example.example", "example.example", tldDb);
checkPublicSuffix("b.example.example", "example.example", tldDb);
checkPublicSuffix("a.b.example.example", "example.example", tldDb);
// Listed, but non-Internet, TLD.
//checkPublicSuffix("local", std::string(), tldDb);
//checkPublicSuffix("example.local", std::string(), tldDb);
//checkPublicSuffix("b.example.local", std::string(), tldDb);
//checkPublicSuffix("a.b.example.local", std::string(), tldDb);
// TLD with only 1 rule.
checkPublicSuffix("biz", std::string(), tldDb);
checkPublicSuffix("domain.biz", "domain.biz", tldDb);
checkPublicSuffix("b.domain.biz", "domain.biz", tldDb);
checkPublicSuffix("a.b.domain.biz", "domain.biz", tldDb);
// TLD with some 2-level rules.
checkPublicSuffix("com", std::string(), tldDb);
checkPublicSuffix("example.com", "example.com", tldDb);
checkPublicSuffix("b.example.com", "example.com", tldDb);
checkPublicSuffix("a.b.example.com", "example.com", tldDb);
checkPublicSuffix("uk.com", std::string(), tldDb);
checkPublicSuffix("example.uk.com", "example.uk.com", tldDb);
checkPublicSuffix("b.example.uk.com", "example.uk.com", tldDb);
checkPublicSuffix("a.b.example.uk.com", "example.uk.com", tldDb);
checkPublicSuffix("test.ac", "test.ac", tldDb);
// TLD with only 1 (wildcard) rule.
checkPublicSuffix("cy", std::string(), tldDb);
checkPublicSuffix("c.cy", std::string(), tldDb);
checkPublicSuffix("b.c.cy", "b.c.cy", tldDb);
checkPublicSuffix("a.b.c.cy", "b.c.cy", tldDb);
// More complex TLD.
checkPublicSuffix("jp", std::string(), tldDb);
checkPublicSuffix("test.jp", "test.jp", tldDb);
checkPublicSuffix("www.test.jp", "test.jp", tldDb);
checkPublicSuffix("ac.jp", std::string(), tldDb);
checkPublicSuffix("test.ac.jp", "test.ac.jp", tldDb);
checkPublicSuffix("www.test.ac.jp", "test.ac.jp", tldDb);
checkPublicSuffix("kyoto.jp", std::string(), tldDb);
checkPublicSuffix("test.kyoto.jp", "test.kyoto.jp", tldDb);
checkPublicSuffix("ide.kyoto.jp", std::string(), tldDb);
checkPublicSuffix("b.ide.kyoto.jp", "b.ide.kyoto.jp", tldDb);
checkPublicSuffix("a.b.ide.kyoto.jp", "b.ide.kyoto.jp", tldDb);
checkPublicSuffix("c.kobe.jp", std::string(), tldDb);
checkPublicSuffix("b.c.kobe.jp", "b.c.kobe.jp", tldDb);
checkPublicSuffix("a.b.c.kobe.jp", "b.c.kobe.jp", tldDb);
checkPublicSuffix("city.kobe.jp", "city.kobe.jp", tldDb);
checkPublicSuffix("www.city.kobe.jp", "city.kobe.jp", tldDb);
// TLD with a wildcard rule and exceptions.
checkPublicSuffix("ck", std::string(), tldDb);
checkPublicSuffix("test.ck", std::string(), tldDb);
checkPublicSuffix("b.test.ck", "b.test.ck", tldDb);
checkPublicSuffix("a.b.test.ck", "b.test.ck", tldDb);
checkPublicSuffix("www.ck", "www.ck", tldDb);
checkPublicSuffix("www.www.ck", "www.ck", tldDb);
// US K12.
checkPublicSuffix("us", std::string(), tldDb);
checkPublicSuffix("test.us", "test.us", tldDb);
checkPublicSuffix("www.test.us", "test.us", tldDb);
checkPublicSuffix("ak.us", std::string(), tldDb);
checkPublicSuffix("test.ak.us", "test.ak.us", tldDb);
checkPublicSuffix("www.test.ak.us", "test.ak.us", tldDb);
checkPublicSuffix("k12.ak.us", std::string(), tldDb);
checkPublicSuffix("test.k12.ak.us", "test.k12.ak.us", tldDb);
checkPublicSuffix("www.test.k12.ak.us", "test.k12.ak.us", tldDb);
// IDN labels.
checkPublicSuffix("食狮.com.cn", "食狮.com.cn", tldDb);
checkPublicSuffix("食狮.公司.cn", "食狮.公司.cn", tldDb);
checkPublicSuffix("www.食狮.公司.cn", "食狮.公司.cn", tldDb);
checkPublicSuffix("shishi.公司.cn", "shishi.公司.cn", tldDb);
checkPublicSuffix("公司.cn", std::string(), tldDb);
checkPublicSuffix("食狮.中国", "食狮.中国", tldDb);
checkPublicSuffix("www.食狮.中国", "食狮.中国", tldDb);
checkPublicSuffix("shishi.中国", "shishi.中国", tldDb);
checkPublicSuffix("中国", std::string(), tldDb);
// Same as above, but punycoded.
// TODO
/*
checkPublicSuffix("xn--85x722f.com.cn", "xn--85x722f.com.cn", tldDb);
checkPublicSuffix("xn--85x722f.xn--55qx5d.cn", "xn--85x722f.xn--55qx5d.cn", tldDb);
checkPublicSuffix("www.xn--85x722f.xn--55qx5d.cn", "xn--85x722f.xn--55qx5d.cn", tldDb);
checkPublicSuffix("shishi.xn--55qx5d.cn", "shishi.xn--55qx5d.cn", tldDb);
checkPublicSuffix("xn--55qx5d.cn", std::string(), tldDb);
checkPublicSuffix("xn--85x722f.xn--fiqs8s", "xn--85x722f.xn--fiqs8s", tldDb);
checkPublicSuffix("www.xn--85x722f.xn--fiqs8s", "xn--85x722f.xn--fiqs8s", tldDb);
checkPublicSuffix("shishi.xn--fiqs8s", "shishi.xn--fiqs8s", tldDb);
checkPublicSuffix("xn--fiqs8s", std::string(), tldDb);
*/
}
