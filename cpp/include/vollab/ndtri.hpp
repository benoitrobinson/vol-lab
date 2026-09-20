// Inverse normal CDF, Cephes ndtri, the same algorithm scipy.special.ndtri uses.
//
// Not bit-portable: the tail branch goes through log(), which is not correctly
// rounded and differs between libms. Roughly 27% of uniforms land there, which
// is why engine parity is asserted with a tolerance on normals and exactly
// only on the uniform bit stream.
#pragma once
#include <cmath>

namespace vollab {

namespace detail {
inline double polevl(double x, const double* c, int n) {
    double ans = c[0];
    for (int i = 1; i <= n; ++i) ans = ans * x + c[i];
    return ans;
}
inline double p1evl(double x, const double* c, int n) {
    double ans = x + c[0];
    for (int i = 1; i < n; ++i) ans = ans * x + c[i];
    return ans;
}

static const double P0[5] = {-5.99633501014107895267E1, 9.80010754185999661536E1,
                             -5.66762857469070293439E1, 1.39312609387279679503E1,
                             -1.23916583867381258016E0};
static const double Q0[8] = {1.95448858338141759834E0, 4.67627912898881538453E0,
                             8.63602421390890590575E1, -2.25462687854119370527E2,
                             2.00260212380060660359E2, -8.20372256168333339912E1,
                             1.59056225126211695515E1, -1.18331621121330003142E0};
static const double P1[9] = {4.05544892305962419923E0, 3.15251094599893866154E1,
                             5.71628192246421288162E1, 4.40805073893200834700E1,
                             1.46849561928858024014E1, 2.18663306850790267539E0,
                             -1.40256079171354495875E-1, -3.50424626827848203418E-2,
                             -8.57456785154685413611E-4};
static const double Q1[8] = {1.57799883256466749731E1, 4.53907635128879210584E1,
                             4.13172038254672030440E1, 1.50425385692907503408E1,
                             2.50464946208309415979E0, -1.42182922854787788574E-1,
                             -3.80806407691578277194E-2, -9.33259480895457427372E-4};
static const double P2[9] = {3.23774891776946035970E0, 6.91522889068984211695E0,
                             3.93881025292474443415E0, 1.33303460815807542389E0,
                             2.01485389549179081538E-1, 1.23716634817820021358E-2,
                             3.01581553508235416007E-4, 2.65806974686737550832E-6,
                             6.23974539184983651783E-9};
static const double Q2[8] = {6.02427039364742014255E0, 3.67983563856160859403E0,
                             1.37702099489081330271E0, 2.16236993594496635890E-1,
                             1.34204006088543189037E-2, 3.28014464682127739104E-4,
                             2.89247864745380683936E-6, 6.79019408009981274425E-9};
constexpr double S2PI = 2.50662827463100050242E0;
}  // namespace detail

inline double ndtri(double y0) {
    using namespace detail;
    if (y0 <= 0.0) return -INFINITY;
    if (y0 >= 1.0) return INFINITY;

    int negate = 1;
    double y = y0;
    if (y > 1.0 - 0.13533528323661269189) { y = 1.0 - y; negate = 0; }

    if (y > 0.13533528323661269189) {
        y = y - 0.5;
        double y2 = y * y;
        double x = y + y * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8));
        return x * S2PI;
    }

    double x = std::sqrt(-2.0 * std::log(y));
    double x0 = x - std::log(x) / x;
    double z = 1.0 / x;
    double x1 = (x < 8.0) ? z * polevl(z, P1, 8) / p1evl(z, Q1, 8)
                          : z * polevl(z, P2, 8) / p1evl(z, Q2, 8);
    x = x0 - x1;
    return negate ? -x : x;
}

}  // namespace vollab
