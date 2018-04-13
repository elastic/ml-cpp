/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CIoManagerTest_h
#define INCLUDED_CIoManagerTest_h

#include <cppunit/extensions/HelperMacros.h>

class CIoManagerTest : public CppUnit::TestFixture
{
    public:
        void testStdinStdout();
        void testFileIoGood();
        void testFileIoBad();
        void testNamedPipeIoGood();
        void testNamedPipeIoBad();

        static CppUnit::Test *suite();

    private:
        void testCommon(const std::string &inputFileName,
                        bool isInputFileNamedPipe,
                        const std::string &outputFileName,
                        bool isOutputFileNamedPipe,
                        bool isGood);
};

#endif // INCLUDED_CIoManagerTest_h

