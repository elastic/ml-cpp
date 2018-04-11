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
#include "CShellArgQuoterTest.h"

#include <core/CLogger.h>
#include <core/CShellArgQuoter.h>

CppUnit::Test* CShellArgQuoterTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CShellArgQuoterTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CShellArgQuoterTest>("CShellArgQuoterTest::testQuote", &CShellArgQuoterTest::testQuote));

    return suiteOfTests;
}

void CShellArgQuoterTest::testQuote() {
    LOG_DEBUG(<< "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("hello")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("\"hello\" there")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("'hello' there")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("hello! there")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this to fail!")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: $HOME")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: %windir%")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: \"$HOME\"")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: \"%windir%\"")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: '$HOME'")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("don't want this expanded: '%windir%'")
              << "\n"
                 "echo "
              << ml::core::CShellArgQuoter::quote("top ^ hat!"));

    // Paste the output of the above into a command prompt and check what
    // happens...
}
