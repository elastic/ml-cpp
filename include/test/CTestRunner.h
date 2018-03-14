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
#ifndef INCLUDED_ml_test_CTestRunner_h
#define INCLUDED_ml_test_CTestRunner_h

#include <test/ImportExport.h>

#include <cppunit/TextTestRunner.h>

#include <string>
#include <vector>


namespace ml {
namespace test {

//! \brief
//! A class to wrap cppunit tests.
//!
//! DESCRIPTION:\n
//! A class to wrap cppunit tests.
//!
//! A cppunit test can be run with the following command line arguments:
//!
//! i) Run all tests, log to stdout:
//!
//! ./test
//!
//! ii) Run named tests, log to stdout:
//!
//! ./test CClass1Test CClass2Test
//!
//! iii) Run all tests, log all except summary to directory 'out_dir'
//! with file name nj_test.<name>, where <name> is the name of the
//! parent parent directory. E.g. for lib/core/testsuite it is
//! out_dir/nj_test.core:
//!
//! ./test -dir out_dir
//!
//! iv) Run named tests, log all except summary to directory 'out_dir'
//! with file name ./test.<name>, where <name> is the name of the
//! parent parent directory. E.g. for lib/core/testsuite it is
//! out_dir/nj_test.core:
//!
//!  ./test -dir out_dir CClass1Test CClass2Test
//!
//! A cppunit Main.cc should be of the form:
//!
//! int
//! main(int argc, const char **argv)
//! {
//!     ml::test::CTestRunner runner(argc, argv);
//!
//!     runner.addTest( CClass1Test::suite() );
//!     runner.addTest( CClass2Test::suite() );
//!     ...
//!
//!     return runner.runTests();
//! }
//!
//! IMPLEMENTATION DECISIONS:\n
//! If a skip file exists in the $CPP_SRC_HOME directory then it is
//! used to cache test results.  This avoids re-running time consuming
//! tests over and over again during a nightly build of multiple products.
//! The skip file should NOT be present in a $CPP_SRC_HOME that is used
//! for interactive development where changes to the code are likely to
//! alter the test results.
//!
class TEST_EXPORT CTestRunner : public CppUnit::TextTestRunner {
    public:
        //! Name of a file storing directories in which tests should be skipped
        //! together with the previous test result
        static const std::string SKIP_FILE_NAME;

        //! Name of file storing results in XML format (to allow display by a
        //! continuous integration system)
        static const std::string XML_RESULT_FILE_NAME;

    public:
        CTestRunner(int argc, const char **argv);
        virtual ~CTestRunner(void);

        //! The command to run tests - DO NOT CALL run()
        virtual bool runTests(void);

    protected:
        //! Time the unit tests
        bool timeTests(const std::string &topPath,
                       const std::string &testPath);

        //! Is the current directory listed in the skip file?  If so, did the
        //! previously run tests pass?
        bool checkSkipFile(const std::string &cwd,
                           bool &passed) const;

        //! Add the current directory to the skip file (if it exists) so that
        //! tests for the same directory aren't re-run.
        bool updateSkipFile(const std::string &cwd,
                            bool passed) const;

    private:
        void processCmdLine(int argc, const char **argv);

    private:
        typedef std::vector<std::string> TStrVec;
        typedef TStrVec::iterator        TStrVecItr;

        TStrVec     m_TestCases;
        std::string m_ExeName;
};


}
}

#endif // INCLUDED_ml_test_CTestRunner_h

