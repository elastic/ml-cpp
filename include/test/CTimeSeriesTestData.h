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

#ifndef INCLUDED_ml_test_CTimeSeriesTestData_h
#define INCLUDED_ml_test_CTimeSeriesTestData_h

#include <core/CoreTypes.h>

#include <test/ImportExport.h>

#include <string>
#include <vector>

namespace ml {
namespace core {
class CRegex;
}
namespace test {

class TEST_EXPORT CTimeSeriesTestData {
public:
    typedef std::vector<double> TDoubleVec;
    typedef TDoubleVec::iterator TDoubleVecItr;
    typedef std::pair<core_t::TTime, double> TTimeDoublePr;
    typedef std::vector<TTimeDoublePr> TTimeDoublePrVec;
    typedef TTimeDoublePrVec::iterator TTimeDoublePrVecItr;
    typedef TTimeDoublePrVec::reverse_iterator TTimeDoublePrVecRItr;
    typedef TTimeDoublePrVec::const_iterator TTimeDoublePrVecCItr;
    typedef std::pair<core_t::TTime, TDoubleVec> TTimeDoubleVecPr;
    typedef std::vector<TTimeDoubleVecPr> TTimeDoubleVecPrVec;

public:
    //! The default regular expression to extract the date
    //! time and value.
    static const std::string DEFAULT_REGEX;
    //! The default regular expression to extract the date
    //! time and bivalued quantities.
    static const std::string DEFAULT_BIVALUED_REGEX;
    //! Empty these don't use strptime.
    static const std::string DEFAULT_DATE_FORMAT;
    //! A regular expression suitable for csv with unix time.
    static const std::string CSV_UNIX_REGEX;
    //! A regular expression suitable for bivalued quantities
    //! for csv fields with unix time.
    static const std::string CSV_UNIX_BIVALUED_REGEX;
    //! Empty these don't use strptime.
    static const std::string CSV_UNIX_DATE_FORMAT;
    //! A regular expression suitable for csv ISO8601 date & time format.
    static const std::string CSV_ISO8601_REGEX;
    //! A regular expression suitable for bivalued quantities
    //! for csv ISO8601 date & time format.
    static const std::string CSV_ISO8601_BIVALUED_REGEX;
    //! The date format for csv ISO8601 date & time format.
    static const std::string CSV_ISO8601_DATE_FORMAT;

public:
    //! Initialise from a text file
    static bool parse(const std::string &fileName,
                      TTimeDoublePrVec &results,
                      const std::string &regex = DEFAULT_REGEX,
                      const std::string &dateFormat = DEFAULT_DATE_FORMAT);

    //! Initialise from a text file (return min/max time)
    static bool parse(const std::string &fileName,
                      TTimeDoublePrVec &results,
                      core_t::TTime &minTime,
                      core_t::TTime &maxTime,
                      const std::string &regex = DEFAULT_REGEX,
                      const std::string &dateFormat = DEFAULT_DATE_FORMAT);

    //! Initialise multivalued from a text file
    static bool parse(const std::string &fileName,
                      TTimeDoubleVecPrVec &results,
                      const std::string &regex = DEFAULT_BIVALUED_REGEX,
                      const std::string &dateFormat = DEFAULT_DATE_FORMAT);

    //! Initialise multivalued from a text file (return min/max time)
    static bool parse(const std::string &fileName,
                      TTimeDoubleVecPrVec &results,
                      core_t::TTime &minTime,
                      core_t::TTime &maxTime,
                      const std::string &regex = DEFAULT_BIVALUED_REGEX,
                      const std::string &dateFormat = DEFAULT_DATE_FORMAT);

    //! Initialise from a text file and parse to counter
    static bool parseCounter(const std::string &fileName, TTimeDoublePrVec &results);

    //! Transform results just to 'value'
    static void transform(const TTimeDoublePrVec &data, TDoubleVec &results);

    //! 1st derivative
    static void derive(const TTimeDoublePrVec &data, TTimeDoublePrVec &results);

    //! Pad a vector from minTime to maxTime with zeros
    static bool pad(const TTimeDoublePrVec &data,
                    core_t::TTime minTime,
                    core_t::TTime maxTime,
                    TTimeDoublePrVec &results);

private:
    template <typename T>
    static bool parse(const std::string &fileName,
                      const std::string &regex,
                      const std::string &dateFormat,
                      std::vector<std::pair<core_t::TTime, T>> &results,
                      core_t::TTime &minTime,
                      core_t::TTime &maxTime);

    template <typename T>
    static bool parseLine(const core::CRegex &tokenRegex,
                          const std::string &dateFormat,
                          const std::string &line,
                          std::vector<std::pair<core_t::TTime, T>> &results);

    //! Prevent construction of this static class
    CTimeSeriesTestData(void);
    CTimeSeriesTestData(const CTimeSeriesTestData &);
};
}
}

#endif// INCLUDED_ml_test_CTimeSeriesTestData_h
