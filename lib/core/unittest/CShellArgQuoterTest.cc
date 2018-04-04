/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
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
    LOG_DEBUG("\n"
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
