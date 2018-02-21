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
#ifndef INCLUDED_CMapPopulationTest_h
#define INCLUDED_CMapPopulationTest_h

#include <cppunit/extensions/HelperMacros.h>

#include <boost/unordered_map.hpp>

#include <map>
#include <string>
#include <vector>


class CMapPopulationTest : public CppUnit::TestFixture
{
    public:
        CMapPopulationTest(void);

        void testMapInsertSpeed(void);

        //! For performance on multi-core hardware, these tests are all run from
        //! the thread pool
        void testMapInsertStr(void);
        void testMapInsertCharP(void);
        void testMapOpSqBracStr(void);
        void testMapOpSqBracCharP(void);
        void testUMapInsertStr(void);
        void testUMapInsertCharP(void);
        void testUMapOpSqBracStr(void);
        void testUMapOpSqBracCharP(void);

        static CppUnit::Test *suite();

        void setUp(void);

    private:
        class CTestData
        {
            public:
                typedef std::vector<std::string>  TStrVec;
                typedef std::vector<const char *> TCharPVec;

            public:
                CTestData(size_t fillSize);

                const TStrVec   &stringKeys(void) const;
                const TStrVec   &stringVals(void) const;
                const TCharPVec &charPtrKeys(void) const;
                const TCharPVec &charPtrVals(void) const;

            private:
                TStrVec   m_StringKeys;
                TStrVec   m_StringVals;

                TCharPVec m_CharPtrKeys;
                TCharPVec m_CharPtrVals;
        };

    private:
        typedef std::map<std::string, std::string>             TStrStrMap;
        typedef std::vector<TStrStrMap>                        TStrStrMapVec;
        typedef boost::unordered_map<std::string, std::string> TStrStrUMap;
        typedef std::vector<TStrStrUMap>                       TStrStrUMapVec;

        template <typename INPUT_CONTAINER, typename MAP_CONTAINER>
        void addInsert(const INPUT_CONTAINER &keys,
                       const INPUT_CONTAINER &values,
                       MAP_CONTAINER &maps) const;

        template <typename INPUT_CONTAINER, typename MAP_CONTAINER>
        void addOpSqBrac(const INPUT_CONTAINER &keys,
                         const INPUT_CONTAINER &values,
                         MAP_CONTAINER &maps) const;

    private:
        static const size_t FILL_SIZE;
        static const size_t TEST_SIZE;

        const CTestData *m_TestData;
};

#endif // INCLUDED_CMapPopulationTest_h

