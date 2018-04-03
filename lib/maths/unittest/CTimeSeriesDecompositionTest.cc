/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include "CTimeSeriesDecompositionTest.h"

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/Constants.h>
#include <core/CRapidXmlParser.h>
#include <core/CRapidXmlStatePersistInserter.h>
#include <core/CRapidXmlStateRestoreTraverser.h>

#include <maths/CDecayRateController.h>
#include <maths/CIntegerTools.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/CSeasonalTime.h>
#include <maths/CTimeSeriesDecomposition.h>

#include <test/CRandomNumbers.h>
#include <test/CTimeSeriesTestData.h>

#include <boost/math/constants/constants.hpp>

#include <fstream>
#include <utility>
#include <vector>

using namespace ml;

namespace
{

using TDoubleDoublePr = std::pair<double, double>;
using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TTimeVec = std::vector<core_t::TTime>;
using TTimeDoublePr = std::pair<core_t::TTime, double>;
using TTimeDoublePrVec = std::vector<TTimeDoublePr>;
using TSeasonalComponentVec = maths_t::TSeasonalComponentVec;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

double mean(const TDoubleDoublePr &x)
{
    return (x.first + x.second) / 2.0;
}

const core_t::TTime FIVE_MINS = 300;
const core_t::TTime TEN_MINS  = 600;
const core_t::TTime HALF_HOUR = core::constants::HOUR / 2;
const core_t::TTime HOUR      = core::constants::HOUR;
const core_t::TTime DAY       = core::constants::DAY;
const core_t::TTime WEEK      = core::constants::WEEK;
const core_t::TTime YEAR      = core::constants::YEAR;

}

void CTimeSeriesDecompositionTest::testSuperpositionOfSines(void)
{
    LOG_DEBUG("+----------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testSuperpositionOfSines  |");
    LOG_DEBUG("+----------------------------------------------------------+");

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 100 * WEEK + 1; time += HALF_HOUR)
    {
        double weekly = 1200.0 + 1000.0 * std::sin(boost::math::double_constants::two_pi
                                                * static_cast<double>(time)
                                                / static_cast<double>(WEEK));
        double daily = 5.0 + 5.0 * std::sin(boost::math::double_constants::two_pi
                                         * static_cast<double>(time)
                                         / static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(weekly * daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 400.0, times.size(), noise);

    core_t::TTime lastWeek = 0;
    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(trend) << ";\n";
    //file << "plot(t, f);\n";
    //TDoubleVec f;
    //TDoubleVec r;

    for (std::size_t i = 0u; i < times.size(); ++i)
    {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);

        if (time >= lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HALF_HOUR)
            {
                TDoubleDoublePr baseline = decomposition.baseline(t, 70.0);
                double residual = std::fabs(trend[t / HALF_HOUR] - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError += std::max(std::max(baseline.first - trend[t / HALF_HOUR],
                                                     trend[t / HALF_HOUR] - baseline.second), 0.0);
                //f.push_back(mean(baseline));
                //r.push_back(mean(baseline) - trend[t / HALF_HOUR]);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG("'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK)
            {
                CPPUNIT_ASSERT(sumResidual < 0.04 * sumValue);
                CPPUNIT_ASSERT(maxResidual < 0.04 * maxValue);
                CPPUNIT_ASSERT(percentileError < 0.02 * sumValue);
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);

    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(r) << ";\n";
    //file << "plot(t(1:length(fe)), fe, 'r');\n";
    //file << "plot(t(1:length(r)), r, 'k');\n";

    CPPUNIT_ASSERT(totalSumResidual < 0.018 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.021 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.01 * totalSumValue);
}

void CTimeSeriesDecompositionTest::testDistortedPeriodic(void)
{
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testDistortedPeriodic  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    const core_t::TTime bucketLength = HOUR;
    const core_t::TTime startTime = 0;
    const TDoubleVec timeseries{
            323444,  960510,  880176,  844190,  823993,  814251,  857187,  856791,  862060,  919632,
            1083704, 2904437, 4601750, 5447896, 5827498, 5924161, 5851895, 5768661, 5927840, 5326236,
            4037245, 1958521, 1360753, 1005194, 901930,  856605,  838370,  810396,  776815,  751163,
            793055,  823974,  820458,  840647,  878594,  1192154, 2321550, 2646460, 2760957, 2838611,
            2784696, 2798327, 2643123, 2028970, 1331199, 1098105, 930971,  907562,  903603,  873554,
            879375,  852853,  828554,  819726,  872418,  856365,  860880,  867119,  873912,  885405,
            1053530, 1487664, 1555301, 1637137, 1672030, 1659346, 1514673, 1228543, 1011740, 928749,
            809702,  838931,  847904,  829188,  822558,  798517,  767446,  750486,  783165,  815612,
            825365,  873486,  1165250, 2977382, 4868975, 6050263, 6470794, 6271899, 6449326, 6352992,
            6162712, 6257295, 4570133, 1781374, 1182546, 665858,  522585,  481588,  395139,  380770,
            379182,  356068,  353498,  347707,  350931,  417253,  989129,  2884728, 4640841, 5423474,
            6246182, 6432793, 6338419, 6312346, 6294323, 6102676, 4505021, 2168289, 1411233, 1055797,
            954338,  918498,  904236,  870193,  843259,  682538,  895407,  883550,  897026,  918838,
            1262303, 3208919, 5193013, 5787263, 6255837, 6337684, 6335017, 6278740, 6191046, 6183259,
            4455055, 2004058, 1425910, 1069949, 942839,  899157,  895133,  858268,  837338,  820983,
            870863,  871873,  881182,  918795,  1237336, 3069272, 4708229, 5672066, 6291124, 6407806,
            6479889, 6533138, 3473382, 6534838, 4800911, 2668073, 1644350, 1282450, 1131734, 1009042,
            891099,  857339,  842849,  816513,  879200,  848292,  858014,  906642,  1208147, 2964568,
            5215885, 5777105, 6332104, 6130733, 6284960, 6157055, 6165520, 5771121, 4309930, 2150044,
            1475275, 1065030, 967267,  890413,  887174,  835741,  814749,  817443,  853085,  851040,
            866029,  867612,  917833,  1225383, 2326451, 2837337, 2975288, 3034415, 3056379, 3181951,
            2938511, 2400202, 1444952, 1058781, 845703,  810419,  805781,  789438,  799674,  775703,
            756145,  727587,  756489,  789886,  784948,  788247,  802013,  832272,  845033,  873396,
            1018788, 1013089, 1095001, 1022910, 798183,  519186,  320507,  247320,  139372,  129477,
            145576,  122348,  120286,  89370,   95583,   88985,   89009,   97425,   103628,  153229,
            675828,  2807240, 4652249, 5170466, 5642965, 5608709, 5697374, 5546758, 5368913, 5161602,
            3793675, 1375703, 593920,  340764,  197075,  174981,  158274,  130148,  125235,  122526,
            113896,  116249,  126881,  213814,  816723,  2690434, 4827493, 5723621, 6219650, 6492638,
            6570160, 6493706, 6495303, 6301872, 4300612, 1543551, 785562,  390012,  234939,  202190,
            142855,  135218,  124238,  111981,  104807,  107687,  129438,  190294,  779698,  2864053,
            5079395, 5912629, 6481437, 6284107, 6451007, 6177724, 5993932, 6075918, 4140658, 1481179,
            682711,  328387,  233915,  182721,  170860,  139540,  137613,  121669,  116906,  121780,
            127887,  199762,  783099,  2890355, 4658524, 5535842, 6117719, 6322938, 6570422, 6396874,
            6586615, 6332100, 4715160, 2604366, 1525620, 906137,  499019,  358856,  225543,  171388,
            153826,  149910,  141092,  136459,  161202,  240704,  766755,  3011958, 5024254, 5901640,
            6244757, 6257553, 6380236, 6394732, 6385424, 5876960, 4182127, 1868461, 883771,  377159,
            264435,  196674,  181845,  138307,  136055,  133143,  129791,  133694,  127502,  136351,
            212305,  777873,  2219051, 2732315, 2965287, 2895288, 2829988, 2818268, 2513817, 1866217,
            985099,  561287,  205195,  173997,  166428,  165294,  130072,  113917,  113282,  112466,
            103406,  115687,  159863,  158310,  225454,  516925,  1268760, 1523357, 1607510, 1560200,
            1483823, 1401526, 999236,  495292,  299905,  286900,  209697,  169881,  157560,  139030,
            132342,  187941,  126162,  106587,  108759,  109495,  116386,  208504,  676794,  1549362,
            2080332, 2488707, 2699237, 2862970, 2602994, 2554047, 2364456, 1997686, 1192434, 891293,
            697769,  391385,  234311,  231839,  160520,  155870,  142220,  139360,  142885,  141589,
            166792,  443202,  2019645, 4558828, 5982111, 6408009, 6514598, 6567566, 6686935, 6532886,
            6473927, 5475257, 2889913, 1524673, 938262,  557410,  325965,  186484,  174831,  211765,
            145477,  148318,  130425,  136431,  182002,  442272,  2078908, 4628945, 5767034, 6212302,
            6566196, 6527687, 6365204, 6226173, 6401203, 5629733, 3004625, 1555528, 1025549, 492910,
            347948,  298725,  272955,  238279,  209290,  188551,  175447,  173960,  190875,  468340,
            1885268, 4133457, 5350137, 5885807, 6331254, 6420279, 6589448, 6483637, 6557769, 5543938,
            3482732, 2010293, 1278681, 735111,  406042,  283694,  181213,  160207,  136347,  113484,
            118521,  127725,  151408,  396552,  1900747, 4400918, 5546984, 6213423, 6464686, 6442904,
            6385002, 6248314, 5880523, 4816342, 2597450, 1374071, 751391,  362615,  215644,  175158,
            116896,  127935,  110407,  113054,  105841,  113717,  177240,  206515,  616005,  1718878,
            2391747, 2450915, 2653897, 2922320, 2808467, 2490078, 1829760, 1219997, 643936,  400743,
            208976,  119623,  110170,  99338,   93661,   100187,  90803,   83980,   75950,   78805,
            95664,   108467,  128293,  294080,  720811,  965705,  1048021, 1125912, 1194746, 1114704,
            799721,  512542,  353694,  291046,  229723,  206109,  183482,  192225,  191906,  176942,
            148163,  145405,  145728,  159016,  181991,  436297,  1983374, 4688246, 5853284, 6243628,
            6730707, 6660743, 6476024, 6422004, 6335113, 5386230, 2761698, 1230646, 763506,  359071,
            223956,  189020,  158090,  145730,  135338,  114941,  108313,  120023,  167161,  440103,
            1781778, 4428615, 5701824, 6296598, 6541586, 6809286, 6716690, 6488941, 6567385, 5633685,
            2760255, 1316495, 732572,  316496,  225013,  202664,  171295,  143195,  123555,  125327,
            123357,  135419,  194933,  428197,  2181096, 4672692, 5854393, 6553263, 6653127, 6772664,
            6899086, 6794041, 6900871, 6087645, 2814928, 1393906, 894417,  413459,  280839,  237468,
            184947,  214658,  180059,  145215,  134793,  133423,  191388,  417885,  2081899, 4836758,
            5803495, 6451696, 7270708, 7628500, 7208066, 7403079, 7548585, 6323024, 3763029, 2197174,
            1359687, 857604,  471729,  338888,  177156,  150619,  145775,  132845,  110888,  121863,
            141321,  440528,  2020529, 4615833, 5772372, 6318037, 6481658, 6454979, 6489447, 6558612,
            6114653, 5009113, 2541519, 1329520, 663124,  311088,  200332,  141768,  120845,  120603,
            114688,  111340,  95757,   91444,   103287,  130905,  551108,  1988083, 2885196, 2962413,
            3070689, 3061746, 2999362, 2993871, 2287683, 1539262, 763592,  393769,  193094,  126535,
            131721,  125761,  105550,  89077,   90295,   93853,   84496,   77731,   89389,   101269,
            153379,  443022,  1114121, 1556021, 1607693, 1589743, 1746231, 1432261, 1022052};

    core_t::TTime time = startTime;
    core_t::TTime lastWeek = startTime;
    maths::CTimeSeriesDecomposition decomposition(0.01, bucketLength);

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    for (std::size_t i = 0u; i < timeseries.size(); ++i, time += bucketLength)
    {
        decomposition.addPoint(time, timeseries[i]);

        if (time >= lastWeek + WEEK || i == boost::size(timeseries) - 1)
        {
            LOG_DEBUG("Processing week");

            //TDoubleVec t;
            //TDoubleVec f;
            //TDoubleVec fe;

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime tt = lastWeek;
                 tt < lastWeek + WEEK &&
                 static_cast<std::size_t>(tt / HOUR) < boost::size(timeseries);
                 tt += HOUR)
            {
                TDoubleDoublePr baseline = decomposition.baseline(tt, 70.0);

                double residual = std::fabs(timeseries[tt / HOUR] - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(timeseries[tt / HOUR]);
                maxValue = std::max(maxValue, std::fabs(timeseries[tt / HOUR]));
                percentileError += std::max(std::max(baseline.first - timeseries[tt / HOUR],
                                                     timeseries[tt / HOUR] - baseline.second), 0.0);

                //t.push_back(tt);
                //f.push_back(timeseries[tt / HOUR]);
                //fe.push_back(mean(baseline));
            }

            LOG_DEBUG("'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG("'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK)
            {
                CPPUNIT_ASSERT(sumResidual < 0.30 * sumValue);
                CPPUNIT_ASSERT(maxResidual < 0.56 * maxValue);
                CPPUNIT_ASSERT(percentileError < 0.21 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            //file << "t = " << core::CContainerPrinter::print(t) << ";\n";
            //file << "f = " << core::CContainerPrinter::print(f) << ";\n";
            //file << "fe = " << core::CContainerPrinter::print(fe) << ";\n";
            //file << "plot(t, f);\n";
            //file << "plot(t, fe, 'r');\n";

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);

    CPPUNIT_ASSERT(totalSumResidual < 0.17 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.23 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.03 * totalSumValue);
}

void CTimeSeriesDecompositionTest::testMinimizeLongComponents(void)
{
    LOG_DEBUG("+------------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testMinimizeLongComponents  |");
    LOG_DEBUG("+------------------------------------------------------------+");

    double weights[] = { 1.0, 0.1, 1.0, 1.0, 0.1, 1.0, 1.0 };

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 100 * WEEK; time += HALF_HOUR)
    {
        double weight = weights[(time / DAY) % 7];
        double daily = 100.0 * std::sin(boost::math::double_constants::two_pi
                                     * static_cast<double>(time)
                                     / static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(weight * daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 16.0, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(trend) << ";\n";
    //file << "plot(t, f);";
    //TDoubleVec f;
    //TDoubleVec r;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;
    double meanSlope = 0.0;
    double refinements = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0u; i < times.size(); ++i)
    {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);

        if (time >= lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HALF_HOUR)
            {
                TDoubleDoublePr baseline = decomposition.baseline(t, 70.0);

                double residual = std::fabs(trend[t / HALF_HOUR] - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError += std::max(std::max(baseline.first - trend[t / HALF_HOUR],
                                                     trend[t / HALF_HOUR] - baseline.second), 0.0);

                //f.push_back(mean(baseline));
                //r.push_back(residual);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG("'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= 2 * WEEK)
            {
                CPPUNIT_ASSERT(sumResidual < 0.16 * sumValue);
                CPPUNIT_ASSERT(maxResidual < 0.35 * maxValue);
                CPPUNIT_ASSERT(percentileError < 0.05 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;

                for (const auto &component : decomposition.seasonalComponents())
                {
                    if (component.initialized() && component.time().period() == WEEK)
                    {
                        double slope = component.valueSpline().absSlope();
                        meanSlope += slope;
                        LOG_DEBUG("weekly |slope| = " << slope);

                        CPPUNIT_ASSERT(slope < 0.0018);
                        refinements += 1.0;
                    }
                }
            }

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);

    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(r) << ";\n";
    //file << "plot(t(1:length(fe)), fe, 'r');\n";
    //file << "plot(t(1:length(r)), r, 'k');\n";

    CPPUNIT_ASSERT(totalSumResidual < 0.06 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.27 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.03 * totalSumValue);

    meanSlope /= refinements;
    LOG_DEBUG("mean weekly |slope| = " << meanSlope);
    CPPUNIT_ASSERT(meanSlope < 0.0015);
}

void CTimeSeriesDecompositionTest::testWeekend(void)
{
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testWeekend  |");
    LOG_DEBUG("+---------------------------------------------+");

    double weights[] = { 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0 };

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 100 * WEEK; time += HALF_HOUR)
    {
        double weight = weights[(time / DAY) % 7];
        double daily = 100.0 * std::sin(boost::math::double_constants::two_pi
                                     * static_cast<double>(time)
                                     / static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(weight * daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 20.0, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(trend) << ";\n";
    //file << "plot(t, f);";
    //TDoubleVec f;
    //TDoubleVec r;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0u; i < times.size(); ++i)
    {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);

        if (time >= lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HALF_HOUR)
            {
                TDoubleDoublePr baseline = decomposition.baseline(t, 70.0);

                double residual = std::fabs(trend[t / HALF_HOUR] - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError += std::max(std::max(baseline.first - trend[t / HALF_HOUR],
                                                     trend[t / HALF_HOUR] - baseline.second), 0.0);

                //f.push_back(mean(baseline));
                //r.push_back(residual);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG("'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= 3 * WEEK)
            {
                CPPUNIT_ASSERT(sumResidual < 0.07 * sumValue);
                CPPUNIT_ASSERT(maxResidual < 0.15 * maxValue);
                CPPUNIT_ASSERT(percentileError < 0.03 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeek += WEEK;
        }
    }

    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(r) << ";\n";
    //file << "plot(t(1:length(fe)), fe, 'r');\n";
    //file << "plot(t(1:length(r)), r, 'k');\n";

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);

    CPPUNIT_ASSERT(totalSumResidual < 0.027 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.12 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.012 * totalSumValue);
}

void CTimeSeriesDecompositionTest::testSinglePeriodicity(void)
{
    LOG_DEBUG("+-------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testSinglePeriodicity  |");
    LOG_DEBUG("+-------------------------------------------------------+");

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 10 * WEEK + 1; time += HALF_HOUR)
    {
        double daily = 100.0 + 100.0 * std::sin(boost::math::double_constants::two_pi
                                             * static_cast<double>(time)
                                             / static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(daily);
    }

    const double noiseMean = 20.0;
    const double noiseVariance = 16.0;
    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(noiseMean, noiseVariance, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(timeseries) << ";\n";
    //file << "plot(t, f);\n";
    //TDoubleVec f;
    //TDoubleVec r;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0u; i < times.size(); ++i)
    {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);

        if (time >= lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (core_t::TTime t = lastWeek;
                 t < lastWeek + WEEK;
                 t += HALF_HOUR)
            {
                TDoubleDoublePr baseline = decomposition.baseline(t, 70.0);

                double residual = std::fabs(trend[t / HALF_HOUR] + noiseMean - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HALF_HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HALF_HOUR]));
                percentileError += std::max(std::max(baseline.first - (trend[t / HALF_HOUR] + noiseMean),
                                                     (trend[t / HALF_HOUR] + noiseMean) - baseline.second), 0.0);

                //f.push_back(mean(baseline));
                //r.push_back(residual);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG("'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= 1 * WEEK)
            {
                CPPUNIT_ASSERT(sumResidual < 0.06 * sumValue);
                CPPUNIT_ASSERT(maxResidual < 0.08 * maxValue);
                CPPUNIT_ASSERT(percentileError < 0.02 * sumValue);

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;

                // Check that only the daily component has been initialized.
                const TSeasonalComponentVec &components = decomposition.seasonalComponents();
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), components.size());
                CPPUNIT_ASSERT_EQUAL(DAY, components[0].time().period());
                CPPUNIT_ASSERT(components[0].initialized());
            }

            lastWeek += WEEK;
        }
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);
    CPPUNIT_ASSERT(totalSumResidual < 0.015 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.042 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.01 * totalSumValue);

    // Check that only the daily component has been initialized.
    const TSeasonalComponentVec &components = decomposition.seasonalComponents();
    CPPUNIT_ASSERT_EQUAL(std::size_t(1), components.size());
    CPPUNIT_ASSERT_EQUAL(DAY, components[0].time().period());
    CPPUNIT_ASSERT(components[0].initialized());

    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(r) << ";\n";
    //file << "plot(t(1:length(fe)), fe, 'r');\n";
    //file << "plot(t(1:length(r)), r, 'k');\n";
}

void CTimeSeriesDecompositionTest::testSeasonalOnset(void)
{
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testSeasonalOnset  |");
    LOG_DEBUG("+---------------------------------------------------+");

    const double daily[] =
        {
             0.0,  0.0,  0.0,  0.0,  5.0,  5.0,
             5.0, 40.0, 40.0, 40.0, 30.0, 30.0,
            35.0, 35.0, 40.0, 50.0, 60.0, 80.0,
            80.0, 10.0,  5.0,  0.0,  0.0,  0.0
        };
    const double weekly[] =
        {
             0.1, 0.1, 1.2, 1.0, 1.0, 0.9, 1.5
        };

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 150 * WEEK + 1; time += HOUR)
    {
        double baseline = 0.0;
        if (time > 10 * WEEK)
        {
            baseline += daily[(time % DAY) / HOUR];
            baseline *= weekly[(time % WEEK) / DAY];
        }
        times.push_back(time);
        trend.push_back(baseline);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 4.0, times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition(0.01, HOUR);

    //std::ofstream file;
    //file.open("results.m");
    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(trend) << ";\n";
    //file << "plot(t, f, 'r');\n";
    //TDoubleVec f;
    //TDoubleVec r;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    core_t::TTime lastWeek = 0;
    for (std::size_t i = 0u; i < times.size(); ++i)
    {
        core_t::TTime time = times[i];
        double value = trend[i] + noise[i];

        decomposition.addPoint(time, value);

        if (time >= lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;
            for (core_t::TTime t = lastWeek; t < lastWeek + WEEK; t += HOUR)
            {
                TDoubleDoublePr baseline = decomposition.baseline(t, 70.0);

                double residual = std::fabs(trend[t / HOUR] - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(trend[t / HOUR]);
                maxValue = std::max(maxValue, std::fabs(trend[t / HOUR]));
                percentileError += std::max(std::max(baseline.first - trend[t / HOUR],
                                                     trend[t / HOUR] - baseline.second), 0.0);
                //f.push_back(mean(baseline));
                //r.push_back(residual);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = "
                      << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
            LOG_DEBUG("'max residual' / 'max value' = "
                      << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));
            LOG_DEBUG("70% error = " << (percentileError == 0.0 ? 0.0 : percentileError / sumValue));

            totalSumResidual += sumResidual;
            totalMaxResidual += maxResidual;
            totalSumValue += sumValue;
            totalMaxValue += maxValue;
            totalPercentileError += percentileError;

            const TSeasonalComponentVec &components = decomposition.seasonalComponents();
            if (time > 11 * WEEK)
            {
                // Check that both components have been initialized.
                CPPUNIT_ASSERT(components.size() > 2);
                CPPUNIT_ASSERT(components[0].initialized());
                CPPUNIT_ASSERT(components[1].initialized());
                CPPUNIT_ASSERT(components[2].initialized());
            }
            else if (time > 10 * WEEK)
            {
                // Check that both components have been initialized.
                CPPUNIT_ASSERT_EQUAL(std::size_t(1), components.size());
                CPPUNIT_ASSERT(components[0].initialized());
            }
            else
            {
                // Check that neither component has been initialized.
                CPPUNIT_ASSERT(components.empty());
            }
            lastWeek += WEEK;
        }
    }

    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(r) << ";\n";
    //file << "plot(t(1:length(fe)), fe);\n";
    //file << "plot(t(1:length(r)), r, 'k');\n";

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);
    CPPUNIT_ASSERT(totalSumResidual < 0.07 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.09 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.03 * totalSumValue);
}

void CTimeSeriesDecompositionTest::testVarianceScale(void)
{
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testVarianceScale  |");
    LOG_DEBUG("+---------------------------------------------------+");

    // Test that variance scales are correctly computed.

    test::CRandomNumbers rng;

    LOG_DEBUG("Variance Spike");
    {
        core_t::TTime time = 0;
        maths::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);

        for (std::size_t i = 0u; i < 50; ++i)
        {
            for (core_t::TTime t = 0; t < DAY; t += TEN_MINS)
            {
                double baseline = 1.0;
                double variance = 1.0;
                if (t >= 3600 && t < 7200)
                {
                    baseline = 5.0;
                    variance = 10.0;
                }
                TDoubleVec value;
                rng.generateNormalSamples(baseline, variance, 1, value);
                decomposition.addPoint(time + t, value[0]);
            }
            time += DAY;
        }

        double meanVariance = (1.0 * 23.0 + 10.0 * 1.0) / 24.0;
        time -= DAY;
        TMeanAccumulator error;
        TMeanAccumulator percentileError;
        TMeanAccumulator meanScale;
        for (core_t::TTime t = 0; t < DAY; t += TEN_MINS)
        {
            double variance = 1.0;
            if (t >= 3600 && t < 7200)
            {
                variance = 10.0;
            }
            double expectedScale = variance / meanVariance;
            TDoubleDoublePr interval = decomposition.scale(time + t, meanVariance, 70.0);
            LOG_DEBUG("time = " << t
                      << ", expectedScale = " << expectedScale
                      << ", scale = " << core::CContainerPrinter::print(interval));
            double scale = (interval.first + interval.second) / 2.0;
            error.add(std::fabs(scale - expectedScale));
            meanScale.add(scale);
            percentileError.add(std::max(std::max(interval.first - expectedScale,
                                         expectedScale - interval.second), 0.0));
        }

        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(error));
        LOG_DEBUG("mean 70% error = " << maths::CBasicStatistics::mean(percentileError))
        LOG_DEBUG("mean scale = " << maths::CBasicStatistics::mean(meanScale));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.29);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(percentileError) < 0.05);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, maths::CBasicStatistics::mean(meanScale), 0.04);
    }
    LOG_DEBUG("Smoothly Varying Variance");
    {
        core_t::TTime time = 0;
        maths::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);

        for (std::size_t i = 0u; i < 50; ++i)
        {
            for (core_t::TTime t = 0; t < DAY; t += TEN_MINS)
            {
                double baseline = 5.0 * std::sin(boost::math::double_constants::two_pi
                                              * static_cast<double>(t)
                                              / static_cast<double>(DAY));
                double variance = 1.0;
                if (t >= 3600 && t < 7200)
                {
                    variance = 10.0;
                }
                TDoubleVec value;
                rng.generateNormalSamples(0.0, variance, 1, value);
                decomposition.addPoint(time + t, baseline + value[0]);
            }
            time += DAY;
        }

        double meanVariance = (1.0 * 23.0 + 10.0 * 1.0) / 24.0;
        time -= DAY;
        TMeanAccumulator error;
        TMeanAccumulator percentileError;
        TMeanAccumulator meanScale;
        for (core_t::TTime t = 0; t < DAY; t += TEN_MINS)
        {
            double variance = 1.0;
            if (t >= 3600 && t < 7200)
            {
                variance = 10.0;
            }
            double expectedScale = variance / meanVariance;
            TDoubleDoublePr interval = decomposition.scale(time + t, meanVariance, 70.0);
            LOG_DEBUG("time = " << t
                      << ", expectedScale = " << expectedScale
                      << ", scale = " << core::CContainerPrinter::print(interval));
            double scale = (interval.first + interval.second) / 2.0;
            error.add(std::fabs(scale - expectedScale));
            meanScale.add(scale);
            percentileError.add(std::max(std::max(interval.first - expectedScale,
                                         expectedScale - interval.second), 0.0));
        }

        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(error));
        LOG_DEBUG("mean 70% error = " << maths::CBasicStatistics::mean(percentileError));
        LOG_DEBUG("mean scale = " << maths::CBasicStatistics::mean(meanScale));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.22);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(percentileError) < 0.1);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, maths::CBasicStatistics::mean(meanScale), 0.01);
    }
    LOG_DEBUG("Long Term Trend");
    {
        const core_t::TTime length = 120 * DAY;

        TTimeVec times;
        TDoubleVec trend;
        for (core_t::TTime time = 0; time < length; time += HALF_HOUR)
        {
            times.push_back(time);
            double x = static_cast<double>(time);
            trend.push_back(150.0 + 100.0 * std::sin(  boost::math::double_constants::two_pi * x
                                                  / static_cast<double>(240 * DAY)
                                                  / (1.0 - x / static_cast<double>(2 * length)))
                                  +  10.0 * std::sin(  boost::math::double_constants::two_pi * x
                                                  / static_cast<double>(DAY)));
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 4.0, times.size(), noise);

        maths::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);
        for (std::size_t i = 0u; i < times.size(); ++i)
        {
            decomposition.addPoint(times[i], trend[i] + 0.3 * noise[i]);
        }

        TMeanAccumulator meanScale;
        double meanVariance = decomposition.meanVariance();
        for (core_t::TTime t = 0; t < DAY; t += TEN_MINS)
        {
            TDoubleDoublePr interval = decomposition.scale(times.back() + t, meanVariance, 70.0);
            LOG_DEBUG("time = " << t
                      << ", scale = " << core::CContainerPrinter::print(interval));
            double scale = (interval.first + interval.second) / 2.0;
            meanScale.add(scale);
        }

        LOG_DEBUG("mean scale = " << maths::CBasicStatistics::mean(meanScale));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, maths::CBasicStatistics::mean(meanScale), 0.01);
    }
}

void CTimeSeriesDecompositionTest::testSpikeyDataProblemCase(void)
{
    LOG_DEBUG("+-----------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testSpikeyDataProblemCase  |");
    LOG_DEBUG("+-----------------------------------------------------------+");

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/spikey_data.csv",
                                                    timeseries,
                                                    startTime,
                                                    endTime,
                                                    "^([0-9]+),([0-9\\.]+)"));
    CPPUNIT_ASSERT(!timeseries.empty());

    LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                timeseries.begin() + 10)
              << " ...");

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    maths::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);
    maths::CNormalMeanPrecConjugate model =
            maths::CNormalMeanPrecConjugate::nonInformativePrior(maths_t::E_ContinuousData, 0.01);

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0u; i < timeseries.size(); ++i)
    {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0u; j < lastWeekTimeseries.size(); ++j)
            {
                TDoubleDoublePr baseline = decomposition.baseline(lastWeekTimeseries[j].first, 70.0);

                double residual = std::fabs(lastWeekTimeseries[j].second - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError += std::max(std::max(baseline.first - lastWeekTimeseries[j].second,
                                                     lastWeekTimeseries[j].second - baseline.second), 0.0);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = "
                      << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
            LOG_DEBUG("'max residual' / 'max value' = "
                      << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= startTime + WEEK)
            {
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeekTimeseries.clear();
            lastWeek += WEEK;
        }
        if (time > lastWeek)
        {
            lastWeekTimeseries.push_back(timeseries[i]);
        }

        if (decomposition.addPoint(time, value))
        {
            model.setToNonInformative(0.0, 0.01);
        }
        model.addSamples(maths_t::TWeightStyleVec{maths_t::E_SampleCountWeight},
                         TDoubleVec{decomposition.detrend(time, value, 70.0)},
                         TDoubleVecVec{TDoubleVec(1, 1.0)});
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);

    CPPUNIT_ASSERT(totalSumResidual < 0.19 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.33 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.14 * totalSumValue);

    //std::ofstream file;
    //file.open("results.m");
    //TTimeVec times;
    //TDoubleVec raw;
    //TDoubleVec baseline;
    //TDoubleVec scales;
    //TDoubleVec probs;

    double pMinScaled = 1.0;
    double pMinUnscaled = 1.0;
    for (std::size_t i = 0u; timeseries[i].first < startTime + DAY; ++i)
    {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;
        double variance = model.marginalLikelihoodVariance();

        double lb, ub;
        maths_t::ETail tail;
        model.probabilityOfLessLikelySamples(
                      maths_t::E_TwoSided,
                      maths_t::TWeightStyleVec{maths_t::E_SampleSeasonalVarianceScaleWeight},
                      TDoubleVec{decomposition.detrend(time, value, 70.0)},
                      TDoubleVecVec{TDoubleVec{std::max(decomposition.scale(time, variance, 70.0).second, 0.25)}},
                      lb, ub, tail);
        double pScaled = (lb + ub) / 2.0;
        pMinScaled = std::min(pMinScaled, pScaled);

        //times.push_back(time);
        //raw.push_back(value);
        //baseline.push_back(mean(decomposition.baseline(time, 70.0)));
        //scales.push_back(mean(decomposition.scale(time, variance, 70.0)));
        //probs.push_back(-std::log(pScaled));

        model.probabilityOfLessLikelySamples(
                      maths_t::E_TwoSided,
                      maths_t::TWeightStyleVec(1, maths_t::E_SampleSeasonalVarianceScaleWeight),
                      TDoubleVec(1, decomposition.detrend(time, value, 70.0)),
                      TDoubleVecVec(1, TDoubleVec(1, 1.0)),
                      lb, ub, tail);
        double pUnscaled = (lb + ub) / 2.0;
        pMinUnscaled = std::min(pMinUnscaled, pUnscaled);
    }

    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(raw) << ";\n";
    //file << "b = " << core::CContainerPrinter::print(baseline) << ";\n";
    //file << "s = " << core::CContainerPrinter::print(scales) << ";\n";
    //file << "p = " << core::CContainerPrinter::print(probs) << ";\n";
    //file << "subplot(3,1,1); hold on; plot(t, r, 'b'); plot(t, b, 'r');\n";
    //file << "subplot(3,1,2); plot(t, s, 'b');\n";
    //file << "subplot(3,1,3); plot(t, p, 'b');\n";

    LOG_DEBUG("pMinScaled = " << pMinScaled);
    LOG_DEBUG("pMinUnscaled = " << pMinUnscaled);
    CPPUNIT_ASSERT(pMinScaled > 1e11 * pMinUnscaled);
}

void CTimeSeriesDecompositionTest::testDiurnalProblemCase(void)
{
    LOG_DEBUG("+--------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testDiurnalProblemCase  |");
    LOG_DEBUG("+--------------------------------------------------------+");

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/diurnal.csv",
                                                    timeseries,
                                                    startTime,
                                                    endTime,
                                                    "^([0-9]+),([0-9\\.]+)"));
    CPPUNIT_ASSERT(!timeseries.empty());

    LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                timeseries.begin() + 10)
              << " ...");

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec times;
    //TDoubleVec values;
    //TDoubleVec f;
    //TDoubleVec r;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    maths::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0u; i < timeseries.size(); ++i)
    {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0u; j < lastWeekTimeseries.size(); ++j)
            {
                TDoubleDoublePr baseline = decomposition.baseline(lastWeekTimeseries[j].first, 70.0);

                double residual = std::fabs(lastWeekTimeseries[j].second - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError += std::max(std::max(baseline.first - lastWeekTimeseries[j].second,
                                                     lastWeekTimeseries[j].second - baseline.second), 0.0);

                //times.push_back(lastWeekTimeseries[j].first);
                //values.push_back(lastWeekTimeseries[j].second);
                //f.push_back(mean(baseline));
                //r.push_back(residual);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = " << sumResidual / sumValue);
            LOG_DEBUG("'max residual' / 'max value' = " << maxResidual / maxValue);
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= startTime + 2 * WEEK)
            {
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeekTimeseries.clear();
            lastWeek += WEEK;
        }
        if (time > lastWeek)
        {
            lastWeekTimeseries.push_back(timeseries[i]);
        }

        decomposition.addPoint(time, value);
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = "
              << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = "
              << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);

    CPPUNIT_ASSERT(totalSumResidual < 0.27 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.72 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.16 * totalSumValue);

    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
    //file << "plot(t, f, 'r');\n";
    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(r) << ";\n";
    //file << "plot(t(1:length(fe)), fe);\n";
    //file << "plot(t(1:length(r)), r, 'k');\n";

    TMeanAccumulator scale;
    double variance = decomposition.meanVariance();
    core_t::TTime time = maths::CIntegerTools::floor(endTime, DAY);
    for (core_t::TTime t = time; t < time + WEEK; t += TEN_MINS)
    {
        scale.add(mean(decomposition.scale(t, variance, 70.0)));
    }

    LOG_DEBUG("scale = " << maths::CBasicStatistics::mean(scale));
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, maths::CBasicStatistics::mean(scale), 0.07);
}

void CTimeSeriesDecompositionTest::testComplexDiurnalProblemCase(void)
{
    LOG_DEBUG("+---------------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testComplexDiurnalProblemCase  |");
    LOG_DEBUG("+---------------------------------------------------------------+");

    TTimeDoublePrVec timeseries;
    core_t::TTime startTime;
    core_t::TTime endTime;
    CPPUNIT_ASSERT(test::CTimeSeriesTestData::parse("testfiles/thirty_minute_samples.csv",
                                                    timeseries,
                                                    startTime,
                                                    endTime,
                                                    test::CTimeSeriesTestData::CSV_ISO8601_REGEX,
                                                    test::CTimeSeriesTestData::CSV_ISO8601_DATE_FORMAT));
    CPPUNIT_ASSERT(!timeseries.empty());

    LOG_DEBUG("timeseries = " << core::CContainerPrinter::print(timeseries.begin(),
                                                                timeseries.begin() + 10)
              << " ...");

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec times;
    //TDoubleVec values;
    //TDoubleVec f;
    //TDoubleVec r;

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    double totalPercentileError = 0.0;

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    core_t::TTime lastWeek = (startTime / WEEK + 1) * WEEK;
    TTimeDoublePrVec lastWeekTimeseries;
    for (std::size_t i = 0u; i < timeseries.size(); ++i)
    {
        core_t::TTime time = timeseries[i].first;
        double value = timeseries[i].second;

        if (time > lastWeek + WEEK)
        {
            LOG_DEBUG("Processing week");

            double sumResidual = 0.0;
            double maxResidual = 0.0;
            double sumValue = 0.0;
            double maxValue = 0.0;
            double percentileError = 0.0;

            for (std::size_t j = 0u; j < lastWeekTimeseries.size(); ++j)
            {
                TDoubleDoublePr baseline = decomposition.baseline(lastWeekTimeseries[j].first, 70.0);

                double residual = std::fabs(lastWeekTimeseries[j].second - mean(baseline));
                sumResidual += residual;
                maxResidual = std::max(maxResidual, residual);
                sumValue += std::fabs(lastWeekTimeseries[j].second);
                maxValue = std::max(maxValue, std::fabs(lastWeekTimeseries[j].second));
                percentileError += std::max(std::max(baseline.first - lastWeekTimeseries[j].second,
                                                     lastWeekTimeseries[j].second - baseline.second), 0.0);

                //times.push_back(lastWeekTimeseries[j].first);
                //values.push_back(lastWeekTimeseries[j].second);
                //f.push_back(mean(baseline));
                //r.push_back(residual);
            }

            LOG_DEBUG("'sum residual' / 'sum value' = "
                      << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
            LOG_DEBUG("'max residual' / 'max value' = "
                      << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));
            LOG_DEBUG("70% error = " << percentileError / sumValue);

            if (time >= startTime + 2 * WEEK)
            {
                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;
                totalPercentileError += percentileError;
            }

            lastWeekTimeseries.clear();
            lastWeek += WEEK;
        }
        if (time > lastWeek)
        {
            lastWeekTimeseries.push_back(timeseries[i]);
        }

        decomposition.addPoint(time, value);
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);
    LOG_DEBUG("total 70% error = " << totalPercentileError / totalSumValue);

    CPPUNIT_ASSERT(totalSumResidual < 0.18 * totalSumValue);
    CPPUNIT_ASSERT(totalMaxResidual < 0.42 * totalMaxValue);
    CPPUNIT_ASSERT(totalPercentileError < 0.08 * totalSumValue);

    //file << "hold on;\n";
    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
    //file << "plot(t, f, 'r');\n";
    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "r = " << core::CContainerPrinter::print(r) << ";\n";
    //file << "plot(t(1:length(fe)), fe);\n";
    //file << "plot(t(1:length(r)), r, 'k');\n";
}

void CTimeSeriesDecompositionTest::testDiurnalPeriodicityWithMissingValues(void)
{
    LOG_DEBUG("+-------------------------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testDiurnalPeriodicityWithMissingValues  |");
    LOG_DEBUG("+-------------------------------------------------------------------------+");

    test::CRandomNumbers rng;

    LOG_DEBUG("Daily Periodic")
    {
        //std::ofstream file;
        //file.open("results.m");
        //TDoubleVec times;
        //TDoubleVec values;
        //TDoubleVec f;

        TMeanAccumulator error;
        maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 50; ++t)
        {
            for (auto value : { 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0})
            {
                if (value > 0.0)
                {
                    TDoubleVec noise;
                    rng.generateNormalSamples(10.0, 2.0, 1, noise);
                    decomposition.addPoint(time, value + noise[0]);
                    if (decomposition.initialized())
                    {
                        error.add(std::fabs(
                             (  value + noise[0]
                              - maths::CBasicStatistics::mean(decomposition.baseline(time, 0.0))))
                            / std::fabs(value + noise[0]));
                    }
                    //times.push_back(time);
                    //values.push_back(value + noise[0]);
                    //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(time, 0.0)));
                }
                time += HALF_HOUR;
            }
        }

        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(error));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.1);

        //file << "hold on;\n";
        //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
        //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
        //file << "plot(t, f, 'r');\n";
        //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
        //file << "plot(t(1:length(fe)), fe);\n";
    }

    LOG_DEBUG("Weekly")
    {
        //std::ofstream file;
        //file.open("results.m");
        //TDoubleVec times;
        //TDoubleVec values;
        //TDoubleVec f;

        TMeanAccumulator error;
        maths::CTimeSeriesDecomposition decomposition(0.01, HOUR);
        core_t::TTime time = 0;
        for (std::size_t t = 0u; t < 10; ++t)
        {
            for (auto value : { 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                               20.0, 18.0, 10.0, 4.0, 4.0, 4.0, 4.0, 5.0, 6.0, 8.0, 9.0, 9.0,
                               10.0, 10.0,  8.0, 4.0, 3.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0,
                                0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0})
            {
                if (value > 0.0)
                {
                    TDoubleVec noise;
                    rng.generateNormalSamples(10.0, 2.0, 1, noise);
                    decomposition.addPoint(time, value + noise[0]);
                    if (decomposition.initialized())
                    {
                        error.add(std::fabs(
                             (  value + noise[0]
                              - maths::CBasicStatistics::mean(decomposition.baseline(time, 0.0))))
                            / std::fabs(value + noise[0]));
                    }
                    //times.push_back(time);
                    //values.push_back(value + noise[0]);
                    //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(time, 0.0)));
                }
                time += HOUR;
            }
        }

        LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(error))
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(error) < 0.1);

        //file << "hold on;\n";
        //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
        //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
        //file << "plot(t, f, 'r');\n";
        //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
        //file << "plot(t(1:length(fe)), fe);\n";
    }
}

void CTimeSeriesDecompositionTest::testLongTermTrend(void)
{
    LOG_DEBUG("+---------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testLongTermTrend  |");
    LOG_DEBUG("+---------------------------------------------------+");

    const core_t::TTime length = 120 * DAY;

    TTimeVec times;
    TDoubleVec trend;

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 25.0, length / HALF_HOUR, noise);

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec f;
    //TDoubleVec values;

    LOG_DEBUG("Linear Ramp")
    {
        for (core_t::TTime time = 0; time < length; time += HALF_HOUR)
        {
            times.push_back(time);
            trend.push_back(5.0 + static_cast<double>(time) / static_cast<double>(DAY));
        }

        maths::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastDay = times[0];

        for (std::size_t i = 0u; i < times.size(); ++i)
        {
            decomposition.addPoint(times[i], trend[i] + noise[i]);

            if (times[i] > lastDay + DAY)
            {
                LOG_DEBUG("Processing day " << times[i] / DAY);

                if (decomposition.initialized())
                {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    TDoubleVec baselines;

                    for (std::size_t j = i - 48; j < i; ++j)
                    {
                        TDoubleDoublePr baseline = decomposition.baseline(times[j], 70.0);
                        baselines.push_back(mean(baseline));
                        double residual = std::fabs(trend[j] - mean(baseline));
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                    }

                    LOG_DEBUG("'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_DEBUG("'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;

                    CPPUNIT_ASSERT(sumResidual / sumValue < 0.05);
                    CPPUNIT_ASSERT(maxResidual / maxValue < 0.05);
                }
                lastDay += DAY;
            }
            //values.push_back(trend[i] + noise[i]);
            //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(times[i])));
        }

        LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
        //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
        //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
        //file << "hold on;\n";
        //file << "plot(t, f, 'r');\n";
        //file << "plot(t, fe);\n";

        CPPUNIT_ASSERT(totalSumResidual / totalSumValue < 0.01);
        CPPUNIT_ASSERT(totalMaxResidual / totalMaxValue < 0.01);
    }

    LOG_DEBUG("Saw Tooth Not Periodic");
    {
        core_t::TTime drops[] =
            {
                0, 30 * DAY, 50 * DAY, 60 * DAY, 85 * DAY, 100 * DAY, 115 * DAY, 120 * DAY
            };

        times.clear();
        trend.clear();

        {
            std::size_t i = 0u;
            for (core_t::TTime time = 0;
                 time < length;
                 time += HALF_HOUR, (time > drops[i] ? ++i : i))
            {
                times.push_back(time);
                trend.push_back(25.0 * static_cast<double>(time - drops[i-1])
                                     / static_cast<double>(drops[i] - drops[i-1] + 1));
            }
        }

        maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastDay = times[0];

        for (std::size_t i = 0u; i < times.size(); ++i)
        {
            decomposition.addPoint(times[i], trend[i] + 0.3 * noise[i]);

            if (times[i] > lastDay + DAY)
            {
                LOG_DEBUG("Processing day " << times[i] / DAY);

                if (decomposition.initialized())
                {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    TDoubleVec baselines;

                    for (std::size_t j = i - 48; j < i; ++j)
                    {
                        TDoubleDoublePr baseline = decomposition.baseline(times[j], 70.0);
                        baselines.push_back(mean(baseline));
                        double residual = std::fabs(trend[j] - mean(baseline));
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                    }

                    LOG_DEBUG("'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_DEBUG("'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;
                }
                lastDay += DAY;
            }
            //values.push_back(trend[i] + 0.3*noise[i]);
            //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(times[i])));
        }

        LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
        //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
        //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
        //file << "hold on;\n";
        //file << "plot(t, f, 'r');\n";
        //file << "plot(t, fe);\n";

        CPPUNIT_ASSERT(totalSumResidual / totalSumValue < 0.38);
        CPPUNIT_ASSERT(totalMaxResidual / totalMaxValue < 0.42);
    }
}

void CTimeSeriesDecompositionTest::testLongTermTrendAndPeriodicity(void)
{
    LOG_DEBUG("+-----------------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testLongTermTrendAndPeriodicity  |");
    LOG_DEBUG("+-----------------------------------------------------------------+");

    // Test long term mean reverting component plus daily periodic component.


    TTimeVec times;
    TDoubleVec trend;
    const core_t::TTime length = 120 * DAY;
    for (core_t::TTime time = 0; time < length; time += HALF_HOUR)
    {
        times.push_back(time);
        double x = static_cast<double>(time);
        trend.push_back(150.0 + 100.0 * std::sin(  boost::math::double_constants::two_pi * x
                                              / static_cast<double>(240 * DAY)
                                              / (1.0 - x / static_cast<double>(2 * length)))
                              +  10.0 * std::sin(  boost::math::double_constants::two_pi * x
                                              / static_cast<double>(DAY)));
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(0.0, 4.0, times.size(), noise);

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec f;
    //TDoubleVec values;

    maths::CTimeSeriesDecomposition decomposition(0.024, HALF_HOUR);

    double totalSumResidual = 0.0;
    double totalMaxResidual = 0.0;
    double totalSumValue = 0.0;
    double totalMaxValue = 0.0;
    core_t::TTime lastDay = times[0];

    for (std::size_t i = 0u; i < times.size(); ++i)
    {
        decomposition.addPoint(times[i], trend[i] + 0.3 * noise[i]);

        if (times[i] > lastDay + DAY)
        {
            LOG_DEBUG("Processing day " << times[i] / DAY);

            if (decomposition.initialized())
            {
                double sumResidual = 0.0;
                double maxResidual = 0.0;
                double sumValue = 0.0;
                double maxValue = 0.0;

                TDoubleVec baselines;

                for (std::size_t j = i - 48; j < i; ++j)
                {
                    TDoubleDoublePr baseline = decomposition.baseline(times[j], 70.0);
                    baselines.push_back(mean(baseline));
                    double residual = std::fabs(trend[j] - mean(baseline));
                    sumResidual += residual;
                    maxResidual = std::max(maxResidual, residual);
                    sumValue += std::fabs(trend[j]);
                    maxValue = std::max(maxValue, std::fabs(trend[j]));
                }

                LOG_DEBUG("'sum residual' / 'sum value' = "
                          << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                LOG_DEBUG("'max residual' / 'max value' = "
                          << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                totalSumResidual += sumResidual;
                totalMaxResidual += maxResidual;
                totalSumValue += sumValue;
                totalMaxValue += maxValue;

                CPPUNIT_ASSERT(sumResidual / sumValue < 0.4);
                CPPUNIT_ASSERT(maxResidual / maxValue < 0.4);
            }
            lastDay += DAY;
        }
        //values.push_back(trend[i] + 0.3 * noise[i]);
        //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(times[i])));
    }

    LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
    LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "plot(t, f, 'r');\n";
    //file << "plot(t, fe);\n";

    CPPUNIT_ASSERT(totalSumResidual / totalSumValue < 0.04);
    CPPUNIT_ASSERT(totalMaxResidual / totalMaxValue < 0.05);
}

void CTimeSeriesDecompositionTest::testNonDiurnal(void)
{
    LOG_DEBUG("+------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testNonDiurnal  |");
    LOG_DEBUG("+------------------------------------------------+");

    test::CRandomNumbers rng;

    LOG_DEBUG("Hourly")
    {
        const core_t::TTime length = 21 * DAY;

        double periodic[]{ 10.0, 1.0, 0.5, 0.5, 1.0, 5.0, 2.0, 1.0, 0.5, 0.5, 1.0, 3.0 };

        TTimeVec times;
        TDoubleVec trends[2]{ TDoubleVec(), TDoubleVec(8 * DAY / FIVE_MINS) };
        for (core_t::TTime time = 0; time < length; time += FIVE_MINS)
        {
            times.push_back(time);
            trends[0].push_back(periodic[(time / FIVE_MINS) % 12]);
            trends[1].push_back(periodic[(time / FIVE_MINS) % 12]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 1.0, trends[1].size(), noise);

        core_t::TTime startTesting[]{3 * HOUR, 16 * DAY};
        TDoubleVec thresholds[]{ TDoubleVec{0.07, 0.06}, TDoubleVec{0.18, 0.13} };

        for (std::size_t t = 0u; t < 2; ++t)
        {
            //std::ofstream file;
            //file.open("results.m");
            //TDoubleVec f;
            //TDoubleVec values;

            maths::CTimeSeriesDecomposition decomposition(0.01, FIVE_MINS);

            double totalSumResidual = 0.0;
            double totalMaxResidual = 0.0;
            double totalSumValue = 0.0;
            double totalMaxValue = 0.0;
            core_t::TTime lastHour = times[0] + 3 * DAY;

            for (std::size_t i = 0u; i < times.size(); ++i)
            {
                decomposition.addPoint(times[i], trends[t][i] + noise[i]);

                if (times[i] > lastHour + HOUR)
                {
                    LOG_DEBUG("Processing hour " << times[i] / HOUR);

                    if (times[i] > startTesting[t])
                    {
                        double sumResidual = 0.0;
                        double maxResidual = 0.0;
                        double sumValue = 0.0;
                        double maxValue = 0.0;

                        TDoubleVec baselines;

                        for (std::size_t j = i - 12; j < i; ++j)
                        {
                            TDoubleDoublePr baseline = decomposition.baseline(times[j], 70.0);
                            baselines.push_back(mean(baseline));
                            double residual = std::fabs(trends[t][j] - mean(baseline));
                            sumResidual += residual;
                            maxResidual = std::max(maxResidual, residual);
                            sumValue += std::fabs(trends[t][j]);
                            maxValue = std::max(maxValue, std::fabs(trends[t][j]));
                        }

                        LOG_DEBUG("'sum residual' / 'sum value' = "
                                  << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                        LOG_DEBUG("'max residual' / 'max value' = "
                                  << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                        totalSumResidual += sumResidual;
                        totalMaxResidual += maxResidual;
                        totalSumValue += sumValue;
                        totalMaxValue += maxValue;

                        CPPUNIT_ASSERT(sumResidual / sumValue < 0.33);
                        CPPUNIT_ASSERT(maxResidual / maxValue < 0.28);
                    }
                    lastHour += HOUR;
                }
                //values.push_back(trends[t][i] + noise[i]);
                //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(times[i])));
            }

            LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
            LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

            //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
            //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
            //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
            //file << "plot(t, f, 'r');\n";
            //file << "plot(t, fe);\n";

            CPPUNIT_ASSERT(totalSumResidual / totalSumValue < thresholds[t][0]);
            CPPUNIT_ASSERT(totalMaxResidual / totalMaxValue < thresholds[t][1]);
        }
    }

    LOG_DEBUG("Two daily");
    {
        const core_t::TTime length = 20 * DAY;

        double periodic[] = { 10.0, 8.0, 5.5, 2.5, 2.0, 5.0, 2.0, 1.0, 1.5, 3.5, 4.0, 7.0 };

        TTimeVec times;
        TDoubleVec trend;
        for (core_t::TTime time = 0; time < length; time += TEN_MINS)
        {
            times.push_back(time);
            trend.push_back(periodic[(time / 4 / HOUR) % 12]);
        }

        TDoubleVec noise;
        rng.generateNormalSamples(0.0, 2.0, times.size(), noise);

        //std::ofstream file;
        //file.open("results.m");
        //TDoubleVec f;
        //TDoubleVec values;

        core_t::TTime startTesting{14 * DAY};
        maths::CTimeSeriesDecomposition decomposition(0.01, TEN_MINS);

        double totalSumResidual = 0.0;
        double totalMaxResidual = 0.0;
        double totalSumValue = 0.0;
        double totalMaxValue = 0.0;
        core_t::TTime lastTwoDay = times[0] + 3 * DAY;

        for (std::size_t i = 0u; i < times.size(); ++i)
        {
            decomposition.addPoint(times[i], trend[i] + noise[i]);

            if (times[i] > lastTwoDay + 2 * DAY)
            {
                LOG_DEBUG("Processing two days " << times[i] / 2 * DAY);

                if (times[i] > startTesting)
                {
                    double sumResidual = 0.0;
                    double maxResidual = 0.0;
                    double sumValue = 0.0;
                    double maxValue = 0.0;

                    TDoubleVec baselines;

                    for (std::size_t j = i - 288; j < i; ++j)
                    {
                        TDoubleDoublePr baseline = decomposition.baseline(times[j], 70.0);
                        baselines.push_back(mean(baseline));
                        double residual = std::fabs(trend[j] - mean(baseline));
                        sumResidual += residual;
                        maxResidual = std::max(maxResidual, residual);
                        sumValue += std::fabs(trend[j]);
                        maxValue = std::max(maxValue, std::fabs(trend[j]));
                    }

                    LOG_DEBUG("'sum residual' / 'sum value' = "
                              << (sumResidual == 0.0 ? 0.0 : sumResidual / sumValue));
                    LOG_DEBUG("'max residual' / 'max value' = "
                              << (maxResidual == 0.0 ? 0.0 : maxResidual / maxValue));

                    totalSumResidual += sumResidual;
                    totalMaxResidual += maxResidual;
                    totalSumValue += sumValue;
                    totalMaxValue += maxValue;

                    CPPUNIT_ASSERT(sumResidual / sumValue < 0.17);
                    CPPUNIT_ASSERT(maxResidual / maxValue < 0.21);
                }
                lastTwoDay += 2 * DAY;
            }
            //values.push_back(trend[i] + noise[i]);
            //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(times[i])));
        }

        LOG_DEBUG("total 'sum residual' / 'sum value' = " << totalSumResidual / totalSumValue);
        LOG_DEBUG("total 'max residual' / 'max value' = " << totalMaxResidual / totalMaxValue);

        //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
        //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
        //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
        //file << "plot(t, f, 'r');\n";
        //file << "plot(t, fe);\n";

        CPPUNIT_ASSERT(totalSumResidual / totalSumValue < 0.1);
        CPPUNIT_ASSERT(totalMaxResidual / totalMaxValue < 0.18);
    }
}

void CTimeSeriesDecompositionTest::testYearly(void)
{
    LOG_DEBUG("+--------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testYearly  |");
    LOG_DEBUG("+--------------------------------------------+");

    using TDouble1Vec = core::CSmallVector<double, 1>;

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition decomposition(0.012, 4 * HOUR);
    maths::CDecayRateController controller(
            maths::CDecayRateController::E_PredictionBias
          | maths::CDecayRateController::E_PredictionErrorIncrease, 1);
    TDoubleVec noise;
    core_t::TTime time = 0;
    for (/**/; time < 4 * YEAR; time += 4 * HOUR)
    {
        double trend =  15.0 * (2.0 + std::sin(  boost::math::double_constants::two_pi
                                            * static_cast<double>(time)
                                            / static_cast<double>(YEAR)))
                      +  7.5 * std::sin(  boost::math::double_constants::two_pi
                                     * static_cast<double>(time)
                                     / static_cast<double>(DAY));
        rng.generateNormalSamples(0.0, 1.0, 1, noise);
        decomposition.addPoint(time, trend + noise[0]);
        if (decomposition.initialized())
        {
            TDouble1Vec prediction{decomposition.mean(time)};
            TDouble1Vec predictionError{decomposition.detrend(time, trend, 0.0)};
            double multiplier{controller.multiplier(
                    prediction, {predictionError}, 4 * HOUR, 1.0, 0.0005)};
            decomposition.decayRate(multiplier * decomposition.decayRate());
        }
    }

    std::ofstream file;
    file.open("results.m");
    TDoubleVec f;
    TTimeVec times;
    TDoubleVec values;

    // Predict over one year and check we get reasonable accuracy.
    TMeanAccumulator meanError;
    for (/**/; time < 5 * YEAR; time += 4 * HOUR)
    {
        double trend =  15.0 * (2.0 + std::sin(  boost::math::double_constants::two_pi
                                            * static_cast<double>(time)
                                            / static_cast<double>(YEAR)))
                      +  7.5 * std::sin(  boost::math::double_constants::two_pi
                                     * static_cast<double>(time)
                                     / static_cast<double>(DAY));
        double prediction = maths::CBasicStatistics::mean(decomposition.baseline(time, 0.0));
        double error = std::fabs((prediction - trend) / trend);
        meanError.add(error);
        times.push_back(time);
        values.push_back(trend);
        f.push_back(prediction);
        if (time / HOUR % 40 == 0)
        {
            LOG_DEBUG("error = " << error);
        }
        CPPUNIT_ASSERT(error < 0.1);
    }

    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "plot(t, f, 'r');\n";
    //file << "plot(t, fe);\n";

    LOG_DEBUG("mean error = " << maths::CBasicStatistics::mean(meanError));
    CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanError) < 0.02);
}

void CTimeSeriesDecompositionTest::testCalendar(void)
{
    LOG_DEBUG("+----------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testCalendar  |");
    LOG_DEBUG("+----------------------------------------------+");

    // Test that we significantly reduce the error on the last Friday of each
    // month after estimating the appropriate component.

    TTimeVec months{ 2505600,  // Fri 30th Jan
                     4924800,  // Fri 27th Feb
                     7344000,  // Fri 27th Mar
                     9763200,  // Fri 24th Apr
                     12787200, // Fri 29th May
                     15206400, // Fri 26th Jun
                     18230400, // Fri 31st Jul
                     18316800 };
    core_t::TTime end = months.back();
    TDoubleVec errors{ 5.0, 15.0, 35.0, 32.0, 25.0, 36.0, 22.0, 12.0, 3.0 };

    auto trend = [&months, &errors](core_t::TTime t)
                 {
                     double result = 20.0 + 10.0 * std::sin(  boost::math::double_constants::two_pi
                                                         * static_cast<double>(t)
                                                         / static_cast<double>(DAY));
                     auto i = std::lower_bound(months.begin(), months.end(), t - DAY);
                     if (   t >= *i + 7200
                         && t <  *i + 7200 + static_cast<core_t::TTime>(errors.size()) * HALF_HOUR)
                     {
                         result += errors[(t - (*i + 7200)) / HALF_HOUR];
                     }
                     return result;
                 };

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition decomposition(0.01, HALF_HOUR);

    //std::ofstream file;
    //file.open("results.m");
    //TDoubleVec f;
    //TDoubleVec times;
    //TDoubleVec values;

    TDoubleVec noise;
    for (core_t::TTime time = 0, count = 0; time < end; time += HALF_HOUR)
    {
        rng.generateNormalSamples(0.0, 4.0, 1, noise);

        decomposition.addPoint(time, trend(time) + noise[0]);

        if (time - DAY == *std::lower_bound(months.begin(), months.end(), time - DAY))
        {
            LOG_DEBUG("*** time = " << time << " ***");

            std::size_t largeErrorCount = 0u;

            for (core_t::TTime time_ = time - DAY; time_ < time; time_ += TEN_MINS)
            {
                double prediction = maths::CBasicStatistics::mean(decomposition.baseline(time_));
                double variance   = 4.0 * maths::CBasicStatistics::mean(decomposition.scale(time_, 4.0, 0.0));
                double actual     = trend(time_);
                if (std::fabs(prediction - actual) / std::sqrt(variance) > 3.0)
                {
                    LOG_DEBUG("  prediction = " << prediction);
                    LOG_DEBUG("  variance   = " << variance);
                    LOG_DEBUG("  trend      = " << trend(time_));
                    ++largeErrorCount;
                }
            }

            LOG_DEBUG("large error count = " << largeErrorCount);
            CPPUNIT_ASSERT(++count > 4 || largeErrorCount > 15);
            CPPUNIT_ASSERT(  count < 5 || largeErrorCount <= 5);
        }

        //times.push_back(time);
        //values.push_back(trend(time) + noise[0]);
        //f.push_back(maths::CBasicStatistics::mean(decomposition.baseline(time, 0.0)));
    }

    //file << "t = " << core::CContainerPrinter::print(times) << ";\n";
    //file << "f = " << core::CContainerPrinter::print(values) << ";\n";
    //file << "fe = " << core::CContainerPrinter::print(f) << ";\n";
    //file << "plot(t, f, 'r');\n";
    //file << "plot(t, fe);\n";
}

void CTimeSeriesDecompositionTest::testConditionOfTrend(void)
{
    LOG_DEBUG("+------------------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testConditionOfTrend  |");
    LOG_DEBUG("+------------------------------------------------------+");

    auto trend = [](core_t::TTime time)
        {
            return std::pow(static_cast<double>(time) / static_cast<double>(WEEK), 2.0);
        };

    const core_t::TTime bucketLength = 6 * HOUR;

    test::CRandomNumbers rng;

    maths::CTimeSeriesDecomposition decomposition(0.0005, bucketLength);
    TDoubleVec noise;
    for (core_t::TTime time = 0; time < 10 * YEAR; time += 6 * HOUR)
    {
        rng.generateNormalSamples(0.0, 3.0, 1, noise);
        decomposition.addPoint(time, trend(time) + noise[0]);
        if (time > 10 * WEEK)
        {
            CPPUNIT_ASSERT(std::fabs(decomposition.detrend(time, trend(time), 0.0)) < 3.0);
        }
    }
}

void CTimeSeriesDecompositionTest::testSwap(void)
{
    LOG_DEBUG("+------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testSwap  |");
    LOG_DEBUG("+------------------------------------------+");

    const double decayRate = 0.01;
    const core_t::TTime bucketLength = HALF_HOUR;

    TTimeVec times;
    TDoubleVec trend1;
    TDoubleVec trend2;
    for (core_t::TTime time = 0; time < 10 * WEEK + 1; time += HALF_HOUR)
    {
        double daily = 15.0 + 10.0 * std::sin(boost::math::double_constants::two_pi
                                           * static_cast<double>(time)
                                           / static_cast<double>(DAY));
        times.push_back(time);
        trend1.push_back(daily);
        trend2.push_back(2.0 * daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(20.0, 16.0, 2 * times.size(), noise);

    maths::CTimeSeriesDecomposition decomposition1(decayRate, bucketLength);
    maths::CTimeSeriesDecomposition decomposition2(2.0 * decayRate, 2 * bucketLength);

    for (std::size_t i = 0u; i < times.size(); i += 2)
    {
        decomposition1.addPoint(times[i], trend1[i] + noise[i]);
        decomposition2.addPoint(times[i], trend2[i] + noise[i+1]);
    }

    uint64_t checksum1 = decomposition1.checksum();
    uint64_t checksum2 = decomposition2.checksum();

    LOG_DEBUG("checksum1 = " << checksum1 << ", checksum2 = " << checksum2);

    decomposition1.swap(decomposition2);

    CPPUNIT_ASSERT_EQUAL(checksum1, decomposition2.checksum());
    CPPUNIT_ASSERT_EQUAL(checksum2, decomposition1.checksum());
}

void CTimeSeriesDecompositionTest::testPersist(void)
{
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testPersist  |");
    LOG_DEBUG("+---------------------------------------------+");

    // Check that serialization is idempotent.
    const double decayRate = 0.01;
    const core_t::TTime bucketLength = HALF_HOUR;

    TTimeVec times;
    TDoubleVec trend;
    for (core_t::TTime time = 0; time < 10 * WEEK + 1; time += HALF_HOUR)
    {
        double daily = 15.0 + 10.0 * std::sin(boost::math::double_constants::two_pi
                                           * static_cast<double>(time)
                                           / static_cast<double>(DAY));
        times.push_back(time);
        trend.push_back(daily);
    }

    test::CRandomNumbers rng;
    TDoubleVec noise;
    rng.generateNormalSamples(20.0, 16.0, times.size(), noise);

    maths::CTimeSeriesDecomposition origDecomposition(decayRate, bucketLength);

    for (std::size_t i = 0u; i < times.size(); ++i)
    {
        origDecomposition.addPoint(times[i], trend[i] + noise[i]);
    }

    std::string origXml;
    {
        ml::core::CRapidXmlStatePersistInserter inserter("root");
        origDecomposition.acceptPersistInserter(inserter);
        inserter.toXml(origXml);
    }

    LOG_TRACE("Decomposition XML representation:\n" << origXml);

    // Restore the XML into a new decomposition
    core::CRapidXmlParser parser;
    CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(origXml));
    core::CRapidXmlStateRestoreTraverser traverser(parser);

    maths::CTimeSeriesDecomposition restoredDecomposition(decayRate + 0.1,
                                                          bucketLength,
                                                          maths::CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE,
                                                          traverser);

    std::string newXml;
    {
        core::CRapidXmlStatePersistInserter inserter("root");
        restoredDecomposition.acceptPersistInserter(inserter);
        inserter.toXml(newXml);
    }
    CPPUNIT_ASSERT_EQUAL(origXml, newXml);
}

void CTimeSeriesDecompositionTest::testUpgrade(void)
{
    LOG_DEBUG("+---------------------------------------------+");
    LOG_DEBUG("|  CTimeSeriesDecompositionTest::testUpgrade  |");
    LOG_DEBUG("+---------------------------------------------+");

    // Check we can validly upgrade existing state.

    using TStrVec = std::vector<std::string>;
    auto load = [](const std::string &name, std::string &result)
        {
            std::ifstream file;
            file.open(name);
            std::stringbuf buf;
            file >> &buf;
            result = buf.str();
        };
    auto stringToPair = [](const std::string &str)
        {
            double first;
            double second;
            std::size_t n{str.find(",")};
            CPPUNIT_ASSERT(n != std::string::npos);
            core::CStringUtils::stringToType(str.substr(0, n), first);
            core::CStringUtils::stringToType(str.substr(n + 1), second);
            return TDoubleDoublePr{first, second};
        };

    std::string empty;

    LOG_DEBUG("*** Seasonal and Calendar Components ***");
    {
        std::string xml;
        load("testfiles/CTimeSeriesDecomposition.6.2.seasonal.state.xml", xml);
        LOG_DEBUG("Saved state size = " << xml.size());

        std::string values;
        load("testfiles/CTimeSeriesDecomposition.6.2.seasonal.expected_values.txt", values);
        LOG_DEBUG("Expected values size = " << values.size());
        TStrVec expectedValues;
        core::CStringUtils::tokenise(";", values, expectedValues, empty);

        std::string scales;
        load("testfiles/CTimeSeriesDecomposition.6.2.seasonal.expected_scales.txt", scales);
        LOG_DEBUG("Expected scales size = " << scales.size());
        TStrVec expectedScales;
        core::CStringUtils::tokenise(";", scales, expectedScales, empty);

        CPPUNIT_ASSERT_EQUAL(expectedValues.size(), expectedScales.size());

        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CTimeSeriesDecomposition decomposition(0.1, HALF_HOUR,
                                                      maths::CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE,
                                                      traverser);

        // Check that the decay rates match and the values and variances
        // predictions match the values obtained from 6.2.

        CPPUNIT_ASSERT_EQUAL(0.01, decomposition.decayRate());

        double meanValue{decomposition.mean(60480000)};
        double meanVariance{decomposition.meanVariance()};
        LOG_DEBUG("restored mean value    = " << meanValue);
        LOG_DEBUG("restored mean variance = " << meanVariance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5994.36, meanValue, 0.005);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(286374.0, meanVariance, 0.5);

        for (core_t::TTime time = 60480000, i = 0;
             i < static_cast<core_t::TTime>(expectedValues.size());
             time += HALF_HOUR, ++i)
        {
            TDoubleDoublePr expectedValue{stringToPair(expectedValues[i])};
            TDoubleDoublePr expectedScale{stringToPair(expectedScales[i])};
            TDoubleDoublePr value{decomposition.baseline(time, 10.0)};
            TDoubleDoublePr scale{decomposition.scale(time, 286374.0, 10.0)};
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedValue.first,
                                         value.first,
                                         0.005 * std::fabs(expectedValue.first));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedValue.second,
                                         value.second,
                                         0.005 * std::fabs(expectedValue.second));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedScale.first,
                                         scale.first,
                                         0.005 * expectedScale.first);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedScale.second,
                                         scale.second,
                                         0.005 * std::max(expectedScale.second, 0.4));
        }
    }

    LOG_DEBUG("*** Trend and Seasonal Components ***");
    {
        std::string xml;
        load("testfiles/CTimeSeriesDecomposition.6.2.trend_and_seasonal.state.xml", xml);
        LOG_DEBUG("Saved state size = " << xml.size());

        std::string values;
        load("testfiles/CTimeSeriesDecomposition.6.2.trend_and_seasonal.expected_values.txt", values);
        LOG_DEBUG("Expected values size = " << values.size());
        TStrVec expectedValues;
        core::CStringUtils::tokenise(";", values, expectedValues, empty);

        std::string scales;
        load("testfiles/CTimeSeriesDecomposition.6.2.trend_and_seasonal.expected_scales.txt", scales);
        LOG_DEBUG("Expected scales size = " << scales.size());
        TStrVec expectedScales;
        core::CStringUtils::tokenise(";", scales, expectedScales, empty);

        CPPUNIT_ASSERT_EQUAL(expectedValues.size(), expectedScales.size());

        core::CRapidXmlParser parser;
        CPPUNIT_ASSERT(parser.parseStringIgnoreCdata(xml));
        core::CRapidXmlStateRestoreTraverser traverser(parser);

        maths::CTimeSeriesDecomposition decomposition(0.1, HALF_HOUR,
                                                      maths::CTimeSeriesDecomposition::DEFAULT_COMPONENT_SIZE,
                                                      traverser);

        // Check that the decay rates match and the values and variances
        // predictions are close to the values obtained from 6.2. We can't
        // update the state exactly in this case so the tolerances in this
        // test are significantly larger.

        CPPUNIT_ASSERT_EQUAL(0.024, decomposition.decayRate());

        double meanValue{decomposition.mean(10366200)};
        double meanVariance{decomposition.meanVariance()};
        LOG_DEBUG("restored mean value    = " << meanValue);
        LOG_DEBUG("restored mean variance = " << meanVariance);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(133.207, meanValue, 4.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(96.1654, meanVariance, 4.0);

        TMeanAccumulator meanValueError;
        TMeanAccumulator meanScaleError;
        for (core_t::TTime time = 10366200, i = 0;
             i < static_cast<core_t::TTime>(expectedValues.size());
             time += HALF_HOUR, ++i)
        {
            TDoubleDoublePr expectedValue{stringToPair(expectedValues[i])};
            TDoubleDoublePr expectedScale{stringToPair(expectedScales[i])};
            TDoubleDoublePr value{decomposition.baseline(time, 10.0)};
            TDoubleDoublePr scale{decomposition.scale(time, 96.1654, 10.0)};
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedValue.first,
                                         value.first,
                                         0.1 * std::fabs(expectedValue.first));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedValue.second,
                                         value.second,
                                         0.1 * std::fabs(expectedValue.second));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedScale.first,
                                         scale.first,
                                         0.3 * expectedScale.first);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedScale.second,
                                         scale.second,
                                         0.3 * expectedScale.second);
            meanValueError.add(  std::fabs(expectedValue.first - value.first)
                               / std::fabs(expectedValue.first));
            meanValueError.add(  std::fabs(expectedValue.second - value.second)
                               / std::fabs(expectedValue.second));
            meanScaleError.add(  std::fabs(expectedScale.first - scale.first)
                               / expectedScale.first);
            meanScaleError.add(  std::fabs(expectedScale.second - scale.second)
                               / expectedScale.second);
        }

        LOG_DEBUG("Mean value error = " << maths::CBasicStatistics::mean(meanValueError));
        LOG_DEBUG("Mean scale error = " << maths::CBasicStatistics::mean(meanScaleError));
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanValueError) < 0.06);
        CPPUNIT_ASSERT(maths::CBasicStatistics::mean(meanScaleError) < 0.07);
    }
}


CppUnit::Test *CTimeSeriesDecompositionTest::suite(void)
{
    CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("CTimeSeriesDecompositionTest");

    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testSuperpositionOfSines",
                                   &CTimeSeriesDecompositionTest::testSuperpositionOfSines) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testDistortedPeriodic",
                                   &CTimeSeriesDecompositionTest::testDistortedPeriodic) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testMinimizeLongComponents",
                                   &CTimeSeriesDecompositionTest::testMinimizeLongComponents) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testWeekend",
                                   &CTimeSeriesDecompositionTest::testWeekend) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testSinglePeriodicity",
                                   &CTimeSeriesDecompositionTest::testSinglePeriodicity) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testSeasonalOnset",
                                   &CTimeSeriesDecompositionTest::testSeasonalOnset) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testVarianceScale",
                                   &CTimeSeriesDecompositionTest::testVarianceScale) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testSpikeyDataProblemCase",
                                   &CTimeSeriesDecompositionTest::testSpikeyDataProblemCase) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testDiurnalProblemCase",
                                   &CTimeSeriesDecompositionTest::testDiurnalProblemCase) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testComplexDiurnalProblemCase",
                                   &CTimeSeriesDecompositionTest::testComplexDiurnalProblemCase) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testDiurnalPeriodicityWithMissingValues",
                                   &CTimeSeriesDecompositionTest::testDiurnalPeriodicityWithMissingValues) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testLongTermTrend",
                                   &CTimeSeriesDecompositionTest::testLongTermTrend) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testLongTermTrendAndPeriodicity",
                                   &CTimeSeriesDecompositionTest::testLongTermTrendAndPeriodicity) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testNonDiurnal",
                                   &CTimeSeriesDecompositionTest::testNonDiurnal) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testYearly",
                                   &CTimeSeriesDecompositionTest::testYearly) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testCalendar",
                                   &CTimeSeriesDecompositionTest::testCalendar) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testConditionOfTrend",
                                   &CTimeSeriesDecompositionTest::testConditionOfTrend) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testSwap",
                                   &CTimeSeriesDecompositionTest::testSwap) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testPersist",
                                   &CTimeSeriesDecompositionTest::testPersist) );
    suiteOfTests->addTest( new CppUnit::TestCaller<CTimeSeriesDecompositionTest>(
                                   "CTimeSeriesDecompositionTest::testUpgrade",
                                   &CTimeSeriesDecompositionTest::testUpgrade) );

    return suiteOfTests;
}
