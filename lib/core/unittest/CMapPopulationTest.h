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

class CMapPopulationTest : public CppUnit::TestFixture {
public:
    CMapPopulationTest();

    void testMapInsertSpeed();

    //! For performance on multi-core hardware, these tests are all run from
    //! the thread pool
    void testMapInsertStr();
    void testMapInsertCharP();
    void testMapOpSqBracStr();
    void testMapOpSqBracCharP();
    void testUMapInsertStr();
    void testUMapInsertCharP();
    void testUMapOpSqBracStr();
    void testUMapOpSqBracCharP();

    static CppUnit::Test* suite();

    void setUp();

private:
    class CTestData {
    public:
        using TStrVec = std::vector<std::string>;
        using TCharPVec = std::vector<const char*>;

    public:
        CTestData(size_t fillSize);

        const TStrVec& stringKeys() const;
        const TStrVec& stringVals() const;
        const TCharPVec& charPtrKeys() const;
        const TCharPVec& charPtrVals() const;

    private:
        TStrVec m_StringKeys;
        TStrVec m_StringVals;

        TCharPVec m_CharPtrKeys;
        TCharPVec m_CharPtrVals;
    };

private:
    using TStrStrMap = std::map<std::string, std::string>;
    using TStrStrMapVec = std::vector<TStrStrMap>;
    using TStrStrUMap = boost::unordered_map<std::string, std::string>;
    using TStrStrUMapVec = std::vector<TStrStrUMap>;

    template<typename INPUT_CONTAINER, typename MAP_CONTAINER>
    void addInsert(const INPUT_CONTAINER& keys,
                   const INPUT_CONTAINER& values,
                   MAP_CONTAINER& maps) const;

    template<typename INPUT_CONTAINER, typename MAP_CONTAINER>
    void addOpSqBrac(const INPUT_CONTAINER& keys,
                     const INPUT_CONTAINER& values,
                     MAP_CONTAINER& maps) const;

private:
    static const size_t FILL_SIZE;
    static const size_t TEST_SIZE;

    const CTestData* m_TestData;
};

#endif // INCLUDED_CMapPopulationTest_h
