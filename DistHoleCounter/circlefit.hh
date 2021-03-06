#pragma once
/**
 * @package de.atwillys.cc.swl
 * @license BSD (simplified)
 * @author Stefan Wilhelm (stfwi)
 *
 * @file circlefit.hh
 * @ccflags
 * @ldflags -lm
 * @platform linux, bsd, windows
 * @standard >= c++98
 *
 * -----------------------------------------------------------------------------
 *
 * Class template containing circle fit algorithms.
 *
 *  - Algebraic: Hyperfit
 *  - Geometric: Levenberg-Marquard
 *
 * Usage:
 *
 *  // 1st possibility: arrays of x/y coords
 *  double x[NUM_POINTS], y[NUM_POINTS];
 *  [...]
 *  circle_fit::circle_t circle = circle_fit::fit(x, y, NUM_POINTS); // <-- Hyperfit
 *  circle_fit::circle_t circle = circle_fit::fit_geometric(x,y,NUM_POINTS); // Levenberg-Marquard
 *  double center_x = circle.x;
 *  double center_y = circle.y;
 *  double r = circle.r;
 *  [...]
 *
 *  // 2nd possibility: vector of points
 *  circle_fit::points_t points;
 *  points.push_back(circle_fit::point_t(X0, Y0));
 *  points.push_back(circle_fit::point_t(X1, Y1));
 *  [...]
 *  circle_fit::circle_t circle = circle_fit::fit(points);
 *
 * Annotations:
 *
 *  - The fit() / fit_geometric() functions have optional arguments (especially for the geometric
 *    fit) to optimise the algorighms.
 *
 *  - The circle_t class has additional verbose output.
 *
 * This implementation is based on "Circular and Linear Regression: Fitting Circles and Lines
 * by Least Squares" (ISBN-10: 143983590X), Nikolai Chernov, | cas.uab.edu.
 *
 * -----------------------------------------------------------------------------
 * +++ BSD license header +++
 * Copyright (c) 2012, Stefan Wilhelm (<cerbero s@atwilly s.de>)
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met: (1) Redistributions
 * of source code must retain the above copyright notice, this list of conditions
 * and the following disclaimer. (2) Redistributions in binary form must reproduce
 * the above copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the distribution.
 * (3) Neither the name of this library nor the names of its contributors may be
 * used to endorse or promote products derived from this software without specific
 * prior written permission. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 * -----------------------------------------------------------------------------
 */
#ifndef SW_CIRCLEFIT_HH
#define SW_CIRCLEFIT_HH

 // <editor-fold desc="preprocessor" defaultstate="collapsed">
#include <cmath>
#include <limits>
#include <vector>
#include <climits>
#include <iostream>
#if defined(__MSC_VER)
#define cf_isfinite _isfinite
#define cf_fabs fabs
#else
#define cf_isfinite std::isfinite
#define cf_fabs fabs
#endif
// </editor-fold>

namespace sw {
	namespace detail {

		/**
		 * @class basic_circle_fit
		 */
		template <typename float_type>
		class basic_circle_fit
		{
		public:

			// <editor-fold desc="circle_t, point_t, points_t" defaultstate="collapsed">

			/**
			 * Describes a resulting circle with additional analysis data.
			 */
			class circle_t
			{
			public:

				inline circle_t() : x(0), y(0), r(0), s(0), g(0), i(0), j(0), ok(true)
				{
					;
				}

				inline circle_t(const circle_t& c) : x(c.x), y(c.y), r(c.r), s(c.s), g(c.g), i(c.i),
					j(c.j), ok(c.ok)
				{
					;
				}

				inline circle_t(float_type x_, float_type y_, float_type r_) : x(x_), y(y_), r(r_), s(0), g(0),
					i(0), j(0), ok(true)
				{
					;
				}

				explicit inline circle_t(bool b) : x(0), y(0), r(0), s(0), g(0), i(0), j(0), ok(b)
				{
					;
				}

				virtual ~circle_t()
				{
					;
				}

				circle_t& operator = (const circle_t& c)
				{
					x = c.x; y = c.y; r = c.r; s = c.s; g = c.g; i = c.i; j = c.j; ok = c.ok; return *this;
				}

				friend std::ostream & operator<< (std::ostream & os, const circle_t& c)
				{
					os << "{x:" << c.x << ", y:" << c.y << ", r:" << c.r
						<< ", ok:" << c.ok << ", i:" << c.i << "}"; return os;
				}

			public:

				float_type x, y, r, s, g;
				int i, j;
				bool ok;
			};

			/**
			 * Basic point structure
			 */
			struct point_t
			{
			public:
				float_type x, y;
				inline point_t() : x(0), y(0) {}
				inline point_t(float_type x_, float_type y_) : x(x_), y(y_) {}
				inline point_t(const point_t &p) : x(p.x), y(p.y) {}
				virtual ~point_t() { ; }
				inline point_t& operator = (const point_t &p) { x = p.x; y = p.y; return *this; }
			};

			/**
			 * Point vector
			 */
			typedef std::vector<point_t> points_t;
			// </editor-fold>

			// <editor-fold desc="data_t" defaultstate="collapsed">
			/**
			 * Data class for C-array compatible, heap allocated point cloud.
			 */
			class data_t
			{
			public:

				data_t() : x_mean(0), y_mean(0)
				{
					;
				}

				virtual ~data_t()
				{
					;
				}

				/**
				 * Reserve memory space for n values. Returns true on success.
				 * @param int n
				 * @return bool
				 */
				inline void reserve(int n)
				{
					x.reserve(n); y.reserve(n);
				}

				/**
				 * Data assignment, std::vector's
				 * @return bool
				 */
				template <typename T>
				inline bool set(const std::vector<T>& x_, const std::vector<T>& y_) {
					clear();
					if (x_.empty() || (x_.size() != y_.size())) return false;
					x = x_; y = y_;
					return true;
				}

				/**
				 * Data assignment, point vector/list
				 * @param const points_t & points
				 * @return bool
				 */
				inline bool set(const points_t & points) {
					if (points.size() == 0) return false;
					reserve(points.size());
					for (typename points_t::const_iterator it = points.begin(); it != points.end(); ++it) {
						x.push_back(it->x); y.push_back(it->y);
					}
					return true;
				}

				/**
				 * Clear the object, free heap memory.
				 */
				inline void clear()
				{
					x_mean = y_mean = 0; x.clear(); y.clear();
				}

				/**
				 * Average calculation over all x / y values.
				 */
				inline void means(void)
				{
					x_mean = y_mean = 0.;
					unsigned n = x.size();
					for (int i = n - 1; i >= 0; --i) { x_mean += x[i]; y_mean += y[i]; }
					x_mean /= n; y_mean /= n;
				}

			public:

				std::vector<float_type> x, y;
				float_type x_mean, y_mean;
			};
			// </editor-fold>

		public:

			// <editor-fold desc="algebraic fit interface" defaultstate="collapsed">
			/**
			 * Fit (input pair vector)
			 * @param const points_t& points
			 * @return circle_t
			 */
			static circle_t fit(const points_t & points)
			{
				data_t data; return (data.set(points)) ? fit(data) : circle_t(false);
			}

			/**
			 * Fits x/y arrays with defined length into a result circle structure.
			 * @param float_type *x
			 * @param float_type* y
			 * @param long n
			 * @return circle_t
			 */
			 //  static circle_t fit(float_type *x, float_type* y, long n)
			 //  { data_t data; return (data.set(n, x, y)) ? fit(data) : circle_t(false); }

			   /**
				* Fits x/y numeric vectors.
				* @param float_type *x
				* @param float_type* y
				* @param long n
				* @return circle_t
				*/
			template <typename T>
			static circle_t fit(const std::vector<T>& x, const std::vector<T>& y)
			{
				data_t data; return  (data.set(x, y)) ? fit(data) : circle_t(false);
			}

			/**
			 * Fits a data set to a resulting 2D circle result, optionally geometric.
			 * @param data_t &data
			 * @return circle_t
			 */
			static circle_t fit(data_t &data)
			{
				circle_t c; circlefit_hyper(data, c); return c;
			}
			// </editor-fold>

		public:

			// <editor-fold desc="geometric fit interface" defaultstate="collapsed">
			/**
			 * Fit (input pair vector)
			 * @param const points2d &points
			 * @return circle_t
			 */
			static circle_t fit_geometric(const points_t &points, float_type lambda = 0.2,
				float_type factor_up = 10., float_type factor_down = 0.04, float_type limit = 1.e+6,
				float_type epsilon = 3.0e-8, int max_iterations = 100)
			{
				data_t data; return (!data.set(points)) ? circle_t(false) : fit_geometric(data, lambda,
					factor_up, factor_down, limit, epsilon, max_iterations);
			}

			/**
			 * Fits x/y numeric vectors.
			 * @param float_type *x
			 * @param float_type* y
			 * @param long n
			 * @return circle_t
			 */
			template <typename T>
			static circle_t fit_geometric(const std::vector<T>& x, const std::vector<T>& y,
				float_type lambda = 0.2, float_type factor_up = 10., float_type factor_down = 0.04,
				float_type limit = 1.e+6, float_type epsilon = 3.0e-8, int max_iterations = 100)
			{
				data_t data; return (!data.set(x, y)) ? circle_t(false) : fit_geometric(data, lambda,
					factor_up, factor_down, limit, epsilon, max_iterations);
			}

			/**
			 * Fits a data set to a resulting 2D circle result.
			 * @param data_t &data
			 * @param float_type geometric_lambda=0.2
			 * @return circle_t
			 */
			static circle_t fit_geometric(data_t &data, float_type lambda = 0.2, float_type factor_up = 10.,
				float_type factor_down = 0.04, float_type limit = 1.e+6, float_type epsilon = 3.0e-8,
				int max_iterations = 100)
			{
				circle_t c;
				circlefit_hyper(data, c);
				circle_t initial_guess = c;
				circlefit_levenberg_marquard(data, c, initial_guess, lambda, factor_up, factor_down, limit,
					epsilon, max_iterations);
				return c;
			}
			// </editor-fold>

		protected:

			// <editor-fold desc="algorithms" defaultstate="collapsed">
#define _1  ((float_type)1.0)
#define _2  ((float_type)2.0)
#define _3  ((float_type)3.0)
#define _4  ((float_type)4.0)
#define _10 ((float_type)10.0)

/**
 * Sqare
 * @param float_type x
 * @return float_type
 */
			static inline float_type sqr(float_type x)
			{
				return x * x;
			}

			/**
			 * Deviation
			 * @param data_t& data
			 * @param circle_t& circle
			 * @return float_type
			 */
			static float_type sigma(data_t& data, circle_t& circle)
			{
				int i, n = data.x.size();
				float_type sum = 0., dx, dy, r;
				float_type LargeCircle = _10, a0, b0, del, s, c, x, y, z, p, t, g, W, Z;
				std::vector<float_type> D(n);
				float_type result = 0;
				if (cf_fabs(circle.x) < LargeCircle && cf_fabs(circle.y) < LargeCircle) {
					for (i = 0; i < n; ++i) {
						dx = data.x[i] - circle.x;
						dy = data.y[i] - circle.y;
						D[i] = sqrt(dx * dx + dy * dy);
						sum += D[i];
					}
					r = sum / n;
					for (sum = 0., i = 0; i < n; ++i) sum += sqr(D[i] - r);
					result = sum / n;
				}
				else {
					// Case of a large circle -> prevent numerical errors
					a0 = circle.x - data.x_mean;
					b0 = circle.y - data.y_mean;
					del = _1 / sqrt(a0 * a0 + b0 * b0);
					s = b0 * del;
					c = a0 * del;
					for (W = Z = 0., i = 0; i < n; ++i) {
						x = data.x[i] - data.x_mean;
						y = data.y[i] - data.y_mean;
						z = x * x + y * y;
						p = x * c + y * s;
						t = del * z - _2 * p;
						g = t / (_1 + sqrt(_1 + del * t));
						W += (z + p * g) / (_2 + del * g);
						Z += z;
					}
					W /= n;
					Z /= n;
					result = Z - W * (_2 + del * del * W);
				}
				return result;
			}

			/**
			 * Hyper ("hyperaccurate") circle fit. It is an algebraic circle fit
			 * with "hyperaccuracy" (with zero essential bias).
			 * A. Al-Sharadqah and N. Chernov, "Error analysis for circle fitting algorithms",
			 * Electronic Journal of Statistics, Vol. 3, pages 886-911, (2009)
			 * Combines the Pratt and Taubin fits to eliminate the essential bias.
			 *
			 * It works well whether data points are sampled along an entire circle or
			 * along a small arc.
			 * Its statistical accuracy is slightly higher than that of the geometric fit
			 * (minimizing geometric distances) and higher than that of the Pratt fit
			 * and Taubin fit.
			 * It provides a very good initial guess for a subsequent geometric fit.
			 *
			 * @param data_t& data
			 * @param circle_t& circle
			 * @param int max_iterations=100
			 * @return void
			 */
			static void circlefit_hyper(data_t& data, circle_t& circle, int max_iterations = 100)
			{
				float_type xi, yi, zi;
				float_type Mz, Mxy, Mxx, Myy, Mxz, Myz, Mzz, cov_xy, var_z;
				float_type A0, A1, A2, A22;
				float_type dy, xnew, x, ynew, y;
				float_type det, x_center, y_center;
				data.means();
				int n = data.x.size();

				// Moments
				Mxx = Myy = Mxy = Mxz = Myz = Mzz = 0.;
				for (int i = 0; i < n; ++i) {
					xi = data.x[i] - data.x_mean; // centered x-coordinates
					yi = data.y[i] - data.y_mean; // centered y-coordinates
					zi = xi * xi + yi * yi;
					Mxy += xi * yi; Mxx += xi * xi;
					Myy += yi * yi; Mxz += xi * zi;
					Myz += yi * zi; Mzz += zi * zi;
				}
				Mxx /= n; Myy /= n;
				Mxy /= n; Mxz /= n;
				Myz /= n; Mzz /= n;

				// Coefficients of the characteristic polynomial
				Mz = Mxx + Myy;
				cov_xy = Mxx * Myy - Mxy * Mxy;
				var_z = Mzz - Mz * Mz;
				A2 = _4 * cov_xy - _3 * Mz * Mz - Mzz;
				A1 = var_z * Mz + _4 * cov_xy * Mz - Mxz * Mxz - Myz * Myz;
				A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - var_z * cov_xy;
				A22 = A2 + A2;

				// Finding the root of the characteristic polynomial using Newton's method starting
				// at x=0. (it is guaranteed to converge to the right root)
				// Usually, 4-6 iterations are enough
				int it;
				for (x = 0., y = A0, it = 0; it < max_iterations; ++it) {
					dy = A1 + x * (A22 + 16. * x * x);
					xnew = x - y / dy;
					if ((xnew == x) || (!cf_isfinite(xnew))) break;
					ynew = A0 + xnew * (A1 + xnew * (A2 + _4 * xnew * xnew));
					if (cf_fabs(ynew) >= cf_fabs(y)) break;
					x = xnew;
					y = ynew;
				}

				// Circle parameters
				det = x * x - x * Mz + cov_xy;
				x_center = (Mxz * (Myy - x) - Myz * Mxy) / det / _2;
				y_center = (Myz * (Mxx - x) - Mxz * Mxy) / det / _2;
				circle.x = x_center + data.x_mean;
				circle.y = y_center + data.y_mean;
				circle.r = sqrt(x_center * x_center + y_center * y_center + Mz - x - x);
				circle.s = sigma(data, circle);
				circle.i = 0;
				circle.j = it;
				circle.ok = true;
			}

			/**
			 * Levenberg-Marquard algorithm for x,y,r
			 *
			 * @param data_t& data
			 * @param circle_t& circle
			 * @param circle_t& initial_guess
			 * @param float_type initial_lambda=1
			 * @param float_type factor_up = 10
			 * @param float_type factor_down = 0.04
			 * @param float_type limit = 1.e+6
			 * @param float_type epsilon = 3.0e-8
			 * @param int max_iterations=100
			 * @return void
			 */
			static void circlefit_levenberg_marquard(data_t& data, circle_t& circle,
				circle_t& initial_guess, float_type initial_lambda = 1, float_type factor_up = 10.,
				float_type factor_down = 0.04, float_type limit = 1.e+6, float_type epsilon = 3.0e-8,
				int max_iterations = 100)
			{
				float_type lambda, dx, dy, ri, u, v;
				float_type Mu, Mv, Muu, Mvv, Muv, Mr, UUl, VVl, Nl, F1, F2, F3, dX, dY, dR;
				float_type G11, G22, G33, G12, G13, G23, D1, D2, D3;
				circle_t curr;
				curr = initial_guess;
				curr.s = sigma(data, curr);
				lambda = initial_lambda;
				circle.i = circle.j = 0;
				int n = data.x.size();
				for (circle.i = 1; circle.i <= max_iterations && circle.j < max_iterations; ++circle.i) {
					circle.x = curr.x;
					circle.y = curr.y;
					circle.r = curr.r;
					circle.s = curr.s;
					circle.g = curr.g;
					Mu = Mv = Muu = Mvv = Muv = Mr = 0.;  // Moments
					for (int i = 0; i < n; ++i) {
						dx = data.x[i] - circle.x;
						dy = data.y[i] - circle.y;
						ri = sqrt(dx * dx + dy * dy);
						u = dx / ri; v = dy / ri;
						Mu += u; Mv += v; Muu += u * u; Mvv += v * v; Muv += u * v; Mr += ri;
					}
					Mu /= n; Mv /= n; Muu /= n; Mvv /= n; Muv /= n; Mr /= n;
					F1 = circle.x + circle.r * Mu - data.x_mean;  // Matrices
					F2 = circle.y + circle.r * Mv - data.y_mean;
					F3 = circle.r - Mr;
					circle.g = curr.g = sqrt(F1 * F1 + F2 * F2 + F3 * F3);
					for (; circle.j < max_iterations; ++circle.j) {
						UUl = Muu + lambda;
						VVl = Mvv + lambda;
						Nl = _1 + lambda;
						G11 = sqrt(UUl); // Cholesly decomposition
						G12 = Muv / G11;
						G13 = Mu / G11;
						G22 = sqrt(VVl - G12 * G12);
						G23 = (Mv - G12 * G13) / G22;
						G33 = sqrt(Nl - G13 * G13 - G23 * G23);
						D1 = F1 / G11;
						D2 = (F2 - G12 * D1) / G22;
						D3 = (F3 - G13 * D1 - G23 * D2) / G33;
						dR = D3 / G33;
						dY = (D2 - G23 * dR) / G22;
						dX = (D1 - G12 * dY - G13 * dR) / G11;
						if ((cf_fabs(dR) + cf_fabs(dX) + cf_fabs(dY)) / (_1 + circle.r) < epsilon) {
							circle.ok = true;
							return;
						}
						curr.x = circle.x - dX; // Parameter update
						curr.y = circle.y - dY;
						if (cf_fabs(curr.x) > limit || cf_fabs(curr.y) > limit) {
							return;
						}
						curr.r = circle.r - dR;
						if (curr.r <= 0.) {
							lambda *= factor_up;
							continue;
						}
						curr.s = sigma(data, curr);
						if (curr.s < circle.s) {
							lambda *= factor_down; // improvement --> next iteration
							break;
						}
						else {
							lambda *= factor_up;
						}
					}
				}
			}
#undef _1
#undef _2
#undef _3
#undef _4
#undef _10
#undef cf_isfinite
#undef cf_fabs
			// </editor-fold>
		};
	}
}

// <editor-fold desc="default specialisation" defaultstate="collapsed">
namespace sw {
	typedef detail::basic_circle_fit<double> circle_fit;
}
// </editor-fold>

#endif#pragma once
