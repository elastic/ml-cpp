/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CShellArgQuoter.h>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(CShellArgQuoterTest)

BOOST_AUTO_TEST_CASE(testQuote) {
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

BOOST_AUTO_TEST_SUITE_END()
