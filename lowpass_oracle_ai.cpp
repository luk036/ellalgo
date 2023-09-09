#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

typedef vector<vector<double>> Arr;
typedef pair<Arr, double> Cut;

Arr ndarray(int rows, int cols) {
    return vector<vector<double>>(rows, vector<double>(cols));
}

class LowpassOracle {
private:
    bool more_alt;
    Arr A;
    int nwpass;
    int nwstop;
    double Lpsq;
    double Upsq;

public:
    LowpassOracle(Arr A, int nwpass, int nwstop, double Lpsq, double Upsq) {
        this->A = A;
        this->nwpass = nwpass;
        this->nwstop = nwstop;
        this->Lpsq = Lpsq;
        this->Upsq = Upsq;
        this->more_alt = true;
        this->count = 0;
    }

    pair<pair<Arr, double>, double> assess_optim(Arr x, double Spsq) {
        int n = x.size();
        this->more_alt = true;

        // case 2, passband constraints
        int N = A.size();
        for (int k = 0; k < this->nwpass; k++) {
            double v = 0;
            for (int j = 0; j < n; j++) {
                v += A[k][j] * x[j];
            }
            if (v > Upsq) {
                Arr g = A[k];
                pair<double, double> f = make_pair(v - Upsq, v - Lpsq);
                return make_pair(make_pair(g, f), 0.0);
            }
            if (v < Lpsq) {
                Arr g = A[k];
                pair<double, double> f = make_pair(-v + Lpsq, -v + Upsq);
                return make_pair(make_pair(g, f), 0.0);
            }
        }

        // case 3, stopband constraint
        double fmax = -INFINITY;
        int imax = 0;
        for (int k = nwstop; k < N; k++) {
            double v = 0;
            for (int j = 0; j < n; j++) {
                v += A[k][j] * x[j];
            }
            if (v > Spsq) {
                Arr g = A[k];
                pair<double, double> f = make_pair(v - Spsq, v);
                return make_pair(make_pair(g, f), 0.0);
            }
            if (v < 0) {
                Arr g = A[k];
                pair<double, double> f = make_pair(-v, -v + Spsq);
                return make_pair(make_pair(g, f), 0.0);
            }
            if (v > fmax) {
                fmax = v;
                imax = k;
            }
        }

        // case 4, nonnegative-real constraint on other frequencies
        for (int k = nwpass; k < nwstop; k++) {
            double v = 0;
            for (int j = 0; j < n; j++) {
                v += A[k][j] * x[j];
            }
            if (v < 0) {
                double f = -v;
                Arr g = A[k];
                return make_pair(make_pair(g, f), 0.0); // single cut
            }
        }

        this->more_alt = false;

        // case 1 (unlikely)
        if (x[0] < 0) {
            Arr g(n, 0.0);
            g[0] = -1.0;
            double f = -x[0];
            return make_pair(make_pair(g, f), 0.0);
        }

        // Begin objective function
        Spsq = fmax;
        pair<double, double> f = make_pair(0.0, fmax);
        Arr g = A[imax];
        return make_pair(make_pair(g, f), Spsq);
    }
};

pair<LowpassOracle, double> create_lowpass_case(int N = 48) {
    double delta0_wpass = 0.025;
    double delta0_wstop = 0.125;
    double delta1 = 20 * log10(1 + delta0_wpass);
    double delta2 = 20 * log10(delta0_wstop);

    int m = 15 * N;
    vector<double> w(m);
    for (int i = 0; i < m; i++) {
        w[i] = i * M_PI / (m - 1);
    }

    Arr An(m, vector<double>(N));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < N; j++) {
            An[i][j] = 2 * cos(w[i] * (j + 1));
        }
    }

    Arr A(m, vector<double>(N + 1));
    for (int i = 0; i < m; i++) {
        A[i][0] = 1.0;
        for (int j = 0; j < N; j++) {
            A[i][j + 1] = An[i][j];
        }
    }

    int nwpass = floor(0.12 * (m - 1)) + 1;
    int nwstop = floor(0.20 * (m - 1)) + 1;

    double Lp = pow(10, -delta1 / 20);
    double Up = pow(10, delta1 / 20);
    double Sp = pow(10, delta2 / 20);

    double Lpsq = Lp * Lp;
    double Upsq = Up * Up;
    double Spsq = Sp * Sp;

    LowpassOracle omega(A, nwpass, nwstop, Lpsq, Upsq);
    return make_pair(omega, Spsq);
}

int main() {
    pair<LowpassOracle, double> result = create_lowpass_case();
    LowpassOracle omega = result.first;
    double Spsq = result.second;

    // Use the omega object and Spsq value in further calculations

    return 0;
}

