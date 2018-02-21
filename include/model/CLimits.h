/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_model_CLimits_h
#define INCLUDED_ml_model_CLimits_h

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <model/ImportExport.h>
#include <model/CResourceMonitor.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/shared_ptr.hpp>

#include <iosfwd>
#include <string>


namespace ml
{
namespace model
{

//! \brief
//! Holds configurable limits for the models.
//!
//! DESCRIPTION:\n
//! Holds limits that prevent Ml custom search commands from
//! taking too long to run or using excessive amounts of memory.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Configuration of Ml's analytics commands is stored in config
//! files which are similar in format to Windows .ini files but 
//! with hash as the comment character instead of semi-colon.
//!
//! Boost's property_tree package can parse such config files, as
//! it accepts either hash or semi-colon as comment characters.
//! Therefore, we use boost::property_tree::ini_parser to load the
//! config file.
//!
//! To decouple the public interface from the config file format,
//! the boost property_tree is copied into separate member
//! variables.
//!
class MODEL_EXPORT CLimits
{
    public:
        //! Default number of events to consume during auto-config
        static const size_t DEFAULT_AUTOCONFIG_EVENTS;

        //! Default maximum number of distinct values of a single field before
        //! analysis of that field will be halted
        static const size_t DEFAULT_ANOMALY_MAX_FIELD_VALUES;

        //! Default maximum number of time buckets to process during anomaly
        //! detection before ceasing to output results
        static const size_t DEFAULT_ANOMALY_MAX_TIME_BUCKETS;

        //! Default number of examples to display in results tables
        static const size_t DEFAULT_RESULTS_MAX_EXAMPLES;

        //! Default threshold for unusual probabilities to be output even if
        //! nothing is anomalous on a whole-system basis
        static const double DEFAULT_RESULTS_UNUSUAL_PROBABILITY_THRESHOLD;

        //! Default memory limit for resource monitor
        static const size_t DEFAULT_MEMORY_LIMIT_MB;

    public:
        //! Default constructor
        CLimits();

        //! Default destructor
        ~CLimits();

        //! Initialise from a config file.  This overwrites current settings
        //! with any found in the config file.  Settings that are not present
        //! in the config file will be reset to their default values.
        bool init(const std::string &configFile);

        //! Access to settings
        size_t autoConfigEvents(void) const;
        size_t anomalyMaxTimeBuckets(void) const;
        size_t maxExamples(void) const;
        double unusualProbabilityThreshold(void) const;
        size_t memoryLimitMB(void) const;

        //! Access to the resource monitor
        CResourceMonitor &resourceMonitor(void);

        //! boost::ini_parser doesn't like UTF-8 ini files that begin with byte
        //! order markers.  This function advances the seek pointer of the
        //! stream over a UTF-8 BOM, but only if one exists.
        static void skipUtf8Bom(std::ifstream &strm);

    private:
        //! Helper method for init().
        template <typename FIELDTYPE>
        static bool processSetting(const boost::property_tree::ptree &propTree,
                                   const std::string &iniPath,
                                   const FIELDTYPE &defaultValue,
                                   FIELDTYPE &value)
        {
            try
            {
                // This get() will throw an exception if the path isn't found
                std::string valueStr(propTree.template get<std::string>(iniPath));

                // Use our own string-to-type conversion, because what's built
                // into the boost::property_tree is too lax
                if (core::CStringUtils::stringToType(valueStr,
                                                     value) == false)
                {
                    LOG_ERROR("Invalid value for setting " << iniPath <<
                              " : " << valueStr);
                    return false;
                }
            }
            catch (boost::property_tree::ptree_error &)
            {
                LOG_DEBUG("Using default value (" << defaultValue <<
                          ") for unspecified setting " << iniPath);
                value = defaultValue;
            }

            return true;
        }

    private:
        //! Number of events to consume during auto-config
        size_t m_AutoConfigEvents;

        //! Maximum number of time buckets to process during anomaly detection
        //! before ceasing to output results
        size_t m_AnomalyMaxTimeBuckets;

        //! How many examples should we display in results tables?
        size_t m_MaxExamples;

        //! Probability threshold for results to be output
        double m_UnusualProbabilityThreshold;

        //! Size of the memory limit for the resource monitor, in MB
        size_t m_MemoryLimitMB;

        //! Resource monitor instance
        CResourceMonitor m_ResourceMonitor;
};


}
}

#endif // INCLUDED_ml_model_CLimits_h

