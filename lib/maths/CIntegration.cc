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

#include <maths/CIntegration.h>

#include <core/CLogger.h>

#include <math.h>

namespace ml {
namespace maths {

const double* CIntegration::CGaussLegendreQuadrature::weights(EOrder order) {
    switch (order) {
    case OrderOne:
        return WEIGHTS1;
    case OrderTwo:
        return WEIGHTS2;
    case OrderThree:
        return WEIGHTS3;
    case OrderFour:
        return WEIGHTS4;
    case OrderFive:
        return WEIGHTS5;
    case OrderSix:
        return WEIGHTS6;
    case OrderSeven:
        return WEIGHTS7;
    case OrderEight:
        return WEIGHTS8;
    case OrderNine:
        return WEIGHTS9;
    case OrderTen:
        return WEIGHTS10;
    }

    LOG_ABORT("Unexpected enumeration value " << order);
}

const double* CIntegration::CGaussLegendreQuadrature::abscissas(EOrder order) {
    switch (order) {
    case OrderOne:
        return ABSCISSAS1;
    case OrderTwo:
        return ABSCISSAS2;
    case OrderThree:
        return ABSCISSAS3;
    case OrderFour:
        return ABSCISSAS4;
    case OrderFive:
        return ABSCISSAS5;
    case OrderSix:
        return ABSCISSAS6;
    case OrderSeven:
        return ABSCISSAS7;
    case OrderEight:
        return ABSCISSAS8;
    case OrderNine:
        return ABSCISSAS9;
    case OrderTen:
        return ABSCISSAS10;
    }

    LOG_ABORT("Unexpected enumeration value " << order);
}

const double CIntegration::CGaussLegendreQuadrature::WEIGHTS1[] = {2.0};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS2[] = {1.0, 1.0};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS3[] = {0.8888888888888888,
                                                                   0.5555555555555556,
                                                                   0.5555555555555556};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS4[] = {0.6521451548625461,
                                                                   0.6521451548625461,
                                                                   0.3478548451374538,
                                                                   0.3478548451374538};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS5[] = {0.5688888888888889,
                                                                   0.4786286704993665,
                                                                   0.4786286704993665,
                                                                   0.2369268850561891,
                                                                   0.2369268850561891};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS6[] = {0.3607615730481386,
                                                                   0.3607615730481386,
                                                                   0.4679139345726910,
                                                                   0.4679139345726910,
                                                                   0.1713244923791704,
                                                                   0.1713244923791704};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS7[] = {0.4179591836734694,
                                                                   0.3818300505051189,
                                                                   0.3818300505051189,
                                                                   0.2797053914892766,
                                                                   0.2797053914892766,
                                                                   0.1294849661688697,
                                                                   0.1294849661688697};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS8[] = {0.3626837833783620,
                                                                   0.3626837833783620,
                                                                   0.3137066458778873,
                                                                   0.3137066458778873,
                                                                   0.2223810344533745,
                                                                   0.2223810344533745,
                                                                   0.1012285362903763,
                                                                   0.1012285362903763};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS9[] = {0.3302393550012598,
                                                                   0.1806481606948574,
                                                                   0.1806481606948574,
                                                                   0.0812743883615744,
                                                                   0.0812743883615744,
                                                                   0.3123470770400029,
                                                                   0.3123470770400029,
                                                                   0.2606106964029354,
                                                                   0.2606106964029354};
const double CIntegration::CGaussLegendreQuadrature::WEIGHTS10[] = {0.2955242247147529,
                                                                    0.2955242247147529,
                                                                    0.2692667193099963,
                                                                    0.2692667193099963,
                                                                    0.2190863625159820,
                                                                    0.2190863625159820,
                                                                    0.1494513491505806,
                                                                    0.1494513491505806,
                                                                    0.0666713443086881,
                                                                    0.0666713443086881};

const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS1[] = {0.0};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS2[] = {-0.5773502691896257,
                                                                     0.5773502691896257};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS3[] = {0.0000000000000000,
                                                                     -0.7745966692414834,
                                                                     0.7745966692414834};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS4[] = {-0.3399810435848563,
                                                                     0.3399810435848563,
                                                                     -0.8611363115940526,
                                                                     0.8611363115940526};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS5[] = {0.0000000000000000,
                                                                     -0.5384693101056831,
                                                                     0.5384693101056831,
                                                                     -0.9061798459386640,
                                                                     0.9061798459386640};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS6[] = {0.6612093864662645,
                                                                     -0.6612093864662645,
                                                                     -0.2386191860831969,
                                                                     0.2386191860831969,
                                                                     -0.9324695142031521,
                                                                     0.9324695142031521};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS7[] = {0.0000000000000000,
                                                                     0.4058451513773972,
                                                                     -0.4058451513773972,
                                                                     -0.7415311855993945,
                                                                     0.7415311855993945,
                                                                     -0.9491079123427585,
                                                                     0.9491079123427585};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS8[] = {-0.1834346424956498,
                                                                     0.1834346424956498,
                                                                     -0.5255324099163290,
                                                                     0.5255324099163290,
                                                                     -0.7966664774136267,
                                                                     0.7966664774136267,
                                                                     -0.9602898564975363,
                                                                     0.9602898564975363};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS9[] = {0.0000000000000000,
                                                                     -0.8360311073266358,
                                                                     0.8360311073266358,
                                                                     -0.9681602395076261,
                                                                     0.9681602395076261,
                                                                     -0.3242534234038089,
                                                                     0.3242534234038089,
                                                                     -0.6133714327005904,
                                                                     0.6133714327005904};
const double CIntegration::CGaussLegendreQuadrature::ABSCISSAS10[] = {-0.1488743389816312,
                                                                      0.1488743389816312,
                                                                      -0.4333953941292472,
                                                                      0.4333953941292472,
                                                                      -0.6794095682990244,
                                                                      0.6794095682990244,
                                                                      -0.8650633666889845,
                                                                      0.8650633666889845,
                                                                      -0.9739065285171717,
                                                                      0.9739065285171717};

core::CFastMutex CIntegration::ms_Mutex;
}
}
