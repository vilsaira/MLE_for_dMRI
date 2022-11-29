/* Rician MLE diffusion and kurtosis tensor estimator by Viljami Sairanen (2016)
Based on algorithm in:
"Liu, Jia, Dario Gasbarra, and Juha Railavo. 
"Fast Estimation of Diffusion Tensors under
Rician noise by the EM algorithm."
Journal of neuroscience methods 257 (2016) : 147 - 158" */

// to convert between single and double precision use following changes:
//  double <-> double
// sqrt( <-> sqrt(
// fabs( <-> fabs(
//  exp( <-> exp(
//  log( <-> log(

#include <math.h>

__device__ size_t calculateGlobalIndex() {
	// Which block are we?
	size_t const globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
	// Which THREAD are we within the block?
	size_t const localthreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
	// How big is each block?
	size_t const threadsPerBlock = blockDim.x*blockDim.y;
	// Which THREAD are we overall?
	return localthreadIdx + globalBlockIndex*threadsPerBlock;
}
__device__  double getBesseli0(double x) {
	double ax, ans, y;
	ax = fabs(x);
	if (ax < 3.75) {
		y = x / 3.75;
		y *= y;
		ans = 1.0 + y*(3.5156229 + y*(3.0899424 + y*(1.2067492 +
			y*(0.2659732 + y*(0.360768e-1 + y*0.45813e-2)))));
		ans *= exp(-ax); // scale by exp(-abs(real(x))); see matlab help for besseli
	}
	else {
		y = 3.75 / ax;
		ans = (1.0 / sqrt(ax)) * // scale by exp(-abs(real(x))); see matlab help for besseli
			(0.39894228 + y * (0.1328592e-1
				+ y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
					+ y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
						+ y * (0.392377e-2)))))))));
	}
	return ans;
}
__device__  double getBesseli1(double x) {
	double ax, ans, y;
	ax = fabs(x);
	if (ax < 3.75) {
		y = x / 3.75;
		y *= y;
		ans = ax * (0.5 + y *(0.87890594 + y *(0.51498869 + y *(0.15084934
			+ y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
		ans *= exp(-ax); // scale by exp(-abs(real(x))); see matlab help for besseli
	}
	else {
		y = 3.75 / ax;
		ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
			- y * 0.420059e-2));
		ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
			+ y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
		ans *= 1.0 / sqrt(ax); // scale by exp(-abs(real(x))); see matlab help for besseli
	}
	return x < 0.0 ? -ans : ans;
}
__device__ double getMax(
	double *arr,
	const unsigned int length,
	size_t const THREAD) {
	double ans;
	ans = arr[THREAD * length];
	for (int i = 1; i < length; i++) {
		if (arr[THREAD * length + i] > ans) {
			ans = arr[THREAD * length + i];
		}
	}
	return ans;
}
__device__  void LUdecomposition(double *a, int n, int *indx, double *vv, size_t const THREAD) {
	int i, imax, j, k;
	double big, dum, sum, temp;

	for (i = 0; i<n; i++) {
		big = 0.0;
		for (j = 0; j<n; j++) {
			temp = fabs(a[THREAD * n * n+ i*n + j]);
			if (temp >= big) {
				big = temp;
			}
		}
		if (big == 0.0) { // Singular matrix can't compute
			big = 1.0e-20;
		}
		vv[THREAD * n + i] = 1.0 / big;
	}
	for (j = 0; j<n; j++) {
		for (i = 0; i<j; i++) {
			sum = a[THREAD * n * n+ i*n + j];
			for (k = 0; k<i; k++) {
				sum -= a[THREAD * n * n+ i*n + k] * a[THREAD * n * n+ k*n + j];
			}
			a[THREAD * n * n+ i*n + j] = sum;
		}
		big = 0.0;
		for (i = j; i<n; i++) {
			sum = a[THREAD * n * n+ i*n + j];
			for (k = 0; k<j; k++) {
				sum -= a[THREAD * n * n+ i*n + k] * a[THREAD * n * n+ k*n + j];
			}
			a[THREAD * n * n+ i*n + j] = sum;
			dum = vv[THREAD * n+ i] * fabs(sum);
			if (dum >= big) {
				big = dum;
				imax = i;
			}
		}
		if (j != imax) {
			for (k = 0; k<n; k++) {
				dum = a[THREAD * n * n+ imax*n + k];
				a[imax*n + k] = a[THREAD * n * n+ j*n + k];
				a[THREAD * n * n+ j*n + k] = dum;
			}
			vv[THREAD * n+ imax] = vv[THREAD * n+ j];
		}
		indx[THREAD * n+ j] = imax;
		if (a[THREAD * n * n+ j*n + j] == 0.0) {
			a[THREAD * n * n+ j*n + j] = 1.0e-20;
		}
		if (j != n) {
			dum = 1.0 / a[THREAD * n * n+ j*n + j];
			for (i = j + 1; i<n; i++) {
				a[THREAD * n * n+ i*n + j] *= dum;
			}
		}
	}
}
__device__  void LUsubstitutions(double *a, int n, int *indx, double *b, size_t const THREAD) {
	int i, ii = 0, ip, j;
	double sum;
	for (i = 0; i<n; i++) {
		ip = indx[(THREAD * n) + i];
		sum = b[(THREAD * n) + ip];
		b[(THREAD * n) + ip] = b[(THREAD * n) + i];
		if (ii != 0) {
			for (j = ii - 1; j<i; j++) {
				sum -= a[(THREAD * n * n) + (i * n) + j] * b[(THREAD * n) + j];
			}
		}
		else if (sum != 0) {
			ii = i + 1;
		}
		b[(THREAD * n) + i] = sum;
	}
	for (i = n - 1; i >= 0; i--) {
		sum = b[(THREAD * n) + i];
		for (j = i + 1; j<n; j++) {
			sum -= a[(THREAD * n * n) + (i * n) + j] * b[(THREAD * n) + j];
		}
		b[(THREAD * n) + i] = sum / a[(THREAD * n * n) + (i * n) + i];
	}
}
__device__ void CholeskyDecomposition(double *a, int n, double *p, size_t const THREAD) {
	int i, j, k;
	double sum;
	for (i = 0; i < n; i++) {
		for (j = i; j < n; j++) {
			sum = a[(THREAD * n * n) + (i*n) + j];
			for (k = i-1; k >= 0; k--) {
				sum -= a[(THREAD * n * n) + (i*n) + k]
					* a[(THREAD * n * n) + (j*n) + k];
			}
			if (i == j) {
				if (sum <= 0.0) {
					sum = 1.0e-20; // Cholesky decomposition failed
				}
				p[THREAD*n + i] = sqrt(sum);
			}
			else {
				a[(THREAD*n*n) + (j*n) + i] = sum / p[THREAD*n + i];
			}
		}
	}
}
__device__ void CholeskyBacksubstitution(double *a, int n, double *p, double *b, double *x, size_t const THREAD) {
	int i, k;
	double sum;
	for (i = 0; i < n; i++) { // Solve Ly=b, storing y in x
		sum = b[THREAD*n + i];
		for (k = i-1; k >= 0; k--) {
			sum -= a[(THREAD*n*n) + (i*n) + k] * x[THREAD*n + k];
		}
		x[THREAD*n + i] = sum / p[THREAD*n + i];
	}
	for (i = n; i >= 0; i--) { // Solve L^(T)x=y
		sum = x[THREAD*n + i];
		for (k = i+1; k < n; k++) {
			sum -= a[(THREAD*n*n) + (k*n) + i] * x[THREAD*n + k];
		}
		x[THREAD*n + i] = sum / p[THREAD*n + i];
	}
}
__device__ void calculateExpZTheta(
	double *expZTheta, 
	double *theta, 
	double *Z,
	const unsigned int nParams, 
	const unsigned int nDWIs,
	size_t const THREAD) {

	for (int i = 0; i < nDWIs; i++) {
		expZTheta[THREAD * nDWIs + i] = 0.0;
		for (int j = 0; j < nParams; j++) {
			expZTheta[THREAD * nDWIs + i] +=
				Z[j * nDWIs + i] * theta[THREAD * nParams + j];
		}
		expZTheta[THREAD * nDWIs + i] = exp(expZTheta[THREAD * nDWIs + i]);
	}

}
__device__ void calculateAB_1(
	double *a,
	double *b,
	double *Y,
	double *expZTheta,
	double *sumYSQ,
	const unsigned int nDWIs,
	size_t const THREAD) {

	a[THREAD] = sumYSQ[THREAD];
	for (int i = 0; i < nDWIs; i++) {
		a[THREAD] += expZTheta[THREAD * nDWIs + i] * expZTheta[THREAD * nDWIs + i];
		b[THREAD * nDWIs + i] = Y[THREAD * nDWIs + i] * expZTheta[THREAD * nDWIs + i];
	}

}
__device__ void calculateAB_2(
	double *a,
	double *b,
	double *Y,
	double *Z,
	double *theta,
	double *SigmaSQ,
	double *expZTheta,
	double *twotau,
	const unsigned int nDWIs,
	const unsigned int nParams,
	size_t const THREAD) {
	// Now indexing for i ranges [0, nDWIs-1] and j ranges [1, nParams] since first nParams is the theta(1)
	a[THREAD] = 0.0;
	for (int i = 0; i < nDWIs; i++) {
		expZTheta[THREAD * nDWIs + i] = 0.0;
		for (int j = 1; j < nParams; j++) {
			expZTheta[THREAD * nDWIs + i] +=
				Z[j * nDWIs + i] * theta[THREAD * nParams + j];
		}
		expZTheta[THREAD * nDWIs + i] = exp(expZTheta[THREAD * nDWIs + i]);
		a[THREAD] += expZTheta[THREAD * nDWIs + i] * expZTheta[THREAD * nDWIs + i];
		b[THREAD * nDWIs + i] = Y[THREAD * nDWIs + i] * expZTheta[THREAD * nDWIs + i];
		twotau[THREAD * nDWIs + i] = b[THREAD * nDWIs + i] * exp(theta[THREAD * nParams+0]) / SigmaSQ[THREAD];
	}
	a[THREAD] = log(a[THREAD]);
}
__device__ void calculateEN(
	double *EN,
	double *twotau,
	const unsigned int nDWIs,
	bool *anyEN,
	size_t const THREAD) {
	anyEN[THREAD] = false;
	for (int i = 0; i < nDWIs; i++) {
		EN[THREAD * nDWIs + i] = 0.5 * twotau[THREAD * nDWIs + i] *
			getBesseli1(twotau[THREAD * nDWIs + i]) /
			getBesseli0(twotau[THREAD * nDWIs + i]);
		if (EN[THREAD * nDWIs + i] > 0.0) {
			anyEN[THREAD] = true;
		}
	}
}
__device__ void calculateZTheta(
	double *c,
	double *ZTheta,
	double *theta,
	double *SigmaSQ,
	double *Z,
	const unsigned int nDWIs,
	const unsigned int nParams,
	size_t const THREAD) {
	// Now indexing for i ranges [0, nDWIs-1] and j ranges [1, nParams] since first nParams is the theta(1)
	c[THREAD] = 2.0 * theta[THREAD * nParams+0] - 
		log(2.0 * SigmaSQ[THREAD]);
	for (int i = 0; i < nDWIs; i++) {
		ZTheta[THREAD * nDWIs + i] = 0.0;
		for (int j = 1; j < nParams; j++) {
			ZTheta[THREAD * nDWIs + i] +=
				Z[j * nDWIs + i] * theta[THREAD * nParams + j];
		}
		ZTheta[THREAD * nDWIs + i] *= 2.0;
		ZTheta[THREAD * nDWIs + i] += c[THREAD];
	}
}
__device__ void calculateLoglikelihood(
	double *loglikelihood,
	double *expo,
	double *ZTheta,
	double *scaling,
	double *expScaling,
	double *EN,
	const unsigned int nDWIs,
	size_t const THREAD) {
	loglikelihood[THREAD] = 0.0;
	for (int i = 0; i < nDWIs; i++) {
		expo[THREAD * nDWIs + i] = exp(ZTheta[THREAD * nDWIs + i] - scaling[THREAD]);
		loglikelihood[THREAD] +=
			EN[THREAD * nDWIs + i] * ZTheta[THREAD * nDWIs + i]
			- expo[THREAD * nDWIs + i] * expScaling[THREAD];
	}
}
__device__ void initializeInformationMatrices(
	double *fisherInformation,
	double *fisherInformation_sym,
	const unsigned int nDeltaParams,
	size_t const THREAD) {
	for (int i = 0; i < nDeltaParams*nDeltaParams; i++) {
		fisherInformation[THREAD * nDeltaParams*nDeltaParams + i] = 0.0;
		fisherInformation_sym[THREAD * nDeltaParams*nDeltaParams + i] = 0.0;
	}
}
__device__ void iterateSigmaSQ(
	double *SigmaSQ,
	double *SigmaSQ0,
	double *tmpdouble,
	double *a,
	double *b,
	double *twotau,
	unsigned int *nIterSigmaSQ,
	unsigned int iterLimitSigmaSQ,
	const double toleranceSigmaSQ,
	const unsigned int nDWIs,
	bool *continueSigmaSQIteration,
	size_t const THREAD) {
	// Should be ok
	continueSigmaSQIteration[THREAD] = true;
	nIterSigmaSQ[THREAD] = 0;
	
	while (continueSigmaSQIteration[THREAD]) {		
		(nIterSigmaSQ[THREAD])++;
		SigmaSQ0[THREAD] = SigmaSQ[THREAD];
		tmpdouble[THREAD] = 0.0;
		for (int i = 0; i < nDWIs; i++) {
			twotau[THREAD * nDWIs + i] = b[THREAD * nDWIs + i] / SigmaSQ[THREAD];
			tmpdouble[THREAD] += twotau[THREAD * nDWIs + i] * 
				getBesseli1(twotau[THREAD * nDWIs + i]) / 
				getBesseli0(twotau[THREAD * nDWIs + i]);
		}
		SigmaSQ[THREAD] = 0.5 * a[THREAD] / ((double)(nDWIs) + tmpdouble[THREAD]);
			
		continueSigmaSQIteration[THREAD] =
			((nIterSigmaSQ[THREAD] < iterLimitSigmaSQ)
				&&
			(fabs(SigmaSQ[THREAD] - SigmaSQ0[THREAD]) > toleranceSigmaSQ));
	}

}
__device__ void iterateS0(
	double *theta,
	double *theta1_old,
	double *SigmaSQ,
	double *a,
	double *b,
	double *twotau,
	unsigned int *nIterS0,
	unsigned int iterLimitS0,
	const double toleranceS0,
	const unsigned int nDWIs,
	const unsigned int nParams,
	bool *continueS0Iteration,
	size_t const THREAD) {
	
	continueS0Iteration[THREAD] = true;
	nIterS0[THREAD] = 0;

	while (continueS0Iteration[THREAD]) {
		nIterS0[THREAD]++;
		// Get initial theta(1) parameter
		theta1_old[THREAD] = theta[THREAD * nParams+0];
		// Calculate new theta(1) parameter
		theta[THREAD * nParams+0] = 0.0;
		for (int i = 0; i < nDWIs; i++) {
			theta[THREAD * nParams+0] += (b[THREAD * nDWIs + i] *
				getBesseli1(twotau[THREAD * nDWIs + i]) /
				getBesseli0(twotau[THREAD * nDWIs + i]));
		}
		theta[THREAD * nParams+0] = log(theta[THREAD * nParams+0]) -a[THREAD];
		// Update twotau for the next iteration step
		for (int i = 0; i < nDWIs; i++) {
			twotau[THREAD * nDWIs + i] = b[THREAD * nDWIs + i] *
				exp(theta[THREAD * nParams+0]) / SigmaSQ[THREAD];
		}
		// Test to end while loop

		continueS0Iteration[THREAD] =
			((nIterS0[THREAD] < iterLimitS0)
				&&
				(fabs((theta[THREAD * nParams + 0] - theta1_old[THREAD]) / theta1_old[THREAD])));
	}
}
__device__ void calculateFisherInformation( 
	double *fisherInformation,
	double *fisherInformation_sym,
	double *Z,
	double *score,
	double *DeltaTheta,
	double *expo,
	double *EN,
	double *expScaling,
	const unsigned int nDWIs,
	const unsigned int nParams,
	const unsigned int nDeltaParams,
	size_t const THREAD) {

	for (int j = 1; j < nParams; j++) {
		score[THREAD * nDeltaParams + j - 1] = 0.0;
		for (int i = 0; i < nDWIs; i++) {
			score[THREAD * nDeltaParams + j - 1] +=
				2.0 * Z[j * nDWIs + i] * (EN[THREAD * nDWIs + i] -
					expo[THREAD * nDWIs + i] * expScaling[THREAD]);
			for (int k = 1; k < nParams; k++) { // range of j and k are [1 to nParams]
				fisherInformation[THREAD * nDeltaParams*nDeltaParams + (j - 1)*nDeltaParams + (k - 1)] +=
					4.0 * Z[j * nDWIs + i] * Z[k * nDWIs + i] * expo[THREAD * nDWIs + i];
				// Symmetrize Fisher Information
				fisherInformation_sym[THREAD * nDeltaParams*nDeltaParams + (j - 1)*nDeltaParams + (k - 1)] =
					(fisherInformation[THREAD * nDeltaParams*nDeltaParams + (j - 1)*nDeltaParams + (k - 1)] +
					 fisherInformation[THREAD * nDeltaParams*nDeltaParams + (j - 1)*nDeltaParams + (k - 1)]) *
					0.5 * expScaling[THREAD];
			}
		}
		DeltaTheta[THREAD * nDeltaParams + j - 1] = score[THREAD *nDeltaParams + j - 1];
	}
	// Make copy of symmetric Fisher information matrix
	for (int i = 0; i < nDeltaParams*nDeltaParams; i++) {
		fisherInformation[THREAD * nDeltaParams * nDeltaParams + i] = fisherInformation_sym[THREAD * nDeltaParams * nDeltaParams + i];
	}
}
__device__ void iterateLoglikelihood(
	int *indx,
	double *score,
	double *vv,
	double *DeltaTheta,
	double *Z,
	double *expo,
	double *theta,
	double *loglikelihood,
	double *loglikelihood_old,
	double *new_theta,
	double *regulatorLambda,
	double *fisherInformation,
	double *fisherInformation_sym,
	double *ZTheta,
	double *c,
	double *scaling,
	double *expScaling,
	double *EN,
	const unsigned int nDWIs,
	const unsigned int nParams,
	const unsigned int nDeltaParams,
	const double regulatorLambda0,
	const double regulatorRescaling,
	unsigned int *nIterLoglikelihood,
	const unsigned int iterLimitLoglikelihood,
	const double toleranceLoglikelihood,
	bool *continueLoglikelihoodIteration,
	size_t const THREAD) {

	nIterLoglikelihood[THREAD] = 0;
	continueLoglikelihoodIteration[THREAD] = true;
	regulatorLambda[THREAD] = regulatorLambda0;
	while (continueLoglikelihoodIteration[THREAD]) {
		nIterLoglikelihood[THREAD]++;
		//loglikelihood_old[THREAD] = loglikelihood[THREAD]; // loglikelihood_old is not supposed to be updated in this loop
		// Initialize DeltaTheta for LUdecomposition & substitutions
		// because X = I\score calculated using LUsubstitutions actually
		// replaces values in score and we don't want to loose that information
		// so we have to save score into DeltaTheta variable
		for (int j = 1; j < nParams; j++) {
			DeltaTheta[THREAD * nDeltaParams + j - 1] = score[THREAD *nDeltaParams + j - 1];
		}
		// Regularize Fisher information matrix with lambda
		for (int i = 0; i < nDeltaParams; i++) {
			fisherInformation[THREAD * nDeltaParams*nDeltaParams + i*nDeltaParams + i] =
				fisherInformation_sym[THREAD * nDeltaParams*nDeltaParams + i*nDeltaParams + i]
				+ regulatorLambda[THREAD];
		}
		// Update regulatorLambda
		regulatorLambda[THREAD] *= regulatorRescaling;
		
		//LUdecomposition(fisherInformation, nDeltaParams, indx, vv, THREAD);
		//LUsubstitutions(fisherInformation, nDeltaParams, indx, DeltaTheta, THREAD);
		CholeskyDecomposition(fisherInformation, nDeltaParams, vv, THREAD);
		CholeskyBacksubstitution(fisherInformation, nDeltaParams, vv, score, DeltaTheta, THREAD);
		//goto THE_END_LOGLIKELIHOOD;
		// Calculate new theta(2:end)
		for (int i = 1; i < nParams; i++) {
			new_theta[THREAD * nDeltaParams + i - 1] =
				theta[THREAD * nParams + i] 
				+ DeltaTheta[THREAD * nDeltaParams + i - 1];
		}
		// Calculate ZTheta based on new_theta
		for (int i = 0; i < nDWIs; i++) {
			ZTheta[THREAD * nDWIs + i] = 0.0;
			for (int j = 1; j < nParams; j++) {
				ZTheta[THREAD * nDWIs + i] +=
					Z[j* nDWIs + i] * new_theta[THREAD * nDeltaParams + j - 1];
			}
			ZTheta[THREAD * nDWIs + i] *= 2.0;
			ZTheta[THREAD * nDWIs + i] += c[THREAD]; // c is based on theta(1) and sigmasq that are constant in this loop
		}
		scaling[THREAD] = getMax(ZTheta, nDWIs, THREAD);
		expScaling[THREAD] = exp(scaling[THREAD]);

		// Calculate new loglikelihood
		// calculateLoglikelihood updates loglikelihood and expo variables
		calculateLoglikelihood(loglikelihood, expo, ZTheta, scaling, expScaling, EN, nDWIs, THREAD);

		// Check if new loglikelihood is NaN, if so more regulation is needed
		// (f != f) is true only if f is NaN (IEEE standard)
		if (loglikelihood[THREAD] != loglikelihood[THREAD]) {
			// loglikelihood is NaN, check only iterations
			continueLoglikelihoodIteration[THREAD] = (nIterLoglikelihood[THREAD] < iterLimitLoglikelihood);
		}
		else {
			continueLoglikelihoodIteration[THREAD] =
				((loglikelihood[THREAD] < loglikelihood_old[THREAD])
					&&
					(nIterLoglikelihood[THREAD] < iterLimitLoglikelihood));
		}
	}
	//THE_END_LOGLIKELIHOOD:
}
__device__ void iterateTheta(
	int *indx,
	double *vv,
	double *theta,
	double *ZTheta,
	double *c,
	double *fisherInformation,
	double *fisherInformation_sym,
	double *score,
	double *Z,
	double *EN,
	double *scaling,
	double *expScaling,
	double *expo,
	double *DeltaTheta,
	double *DeltaThetaScore,
	double *new_theta,
	double *loglikelihood,
	double *loglikelihood_old,
	double *regulatorLambda,
	const double regulatorLambda0,
	const double regulatorRescaling,
	const unsigned int nDWIs,
	const unsigned int nParams,
	const unsigned int nDeltaParams,
	unsigned int *nIterTheta,
	unsigned int *nIterLoglikelihood,
	const unsigned int iterLimitTheta,
	const unsigned int iterLimitLoglikelihood,
	const double toleranceTheta,
	const double toleranceLoglikelihood,
	bool *continueThetaIteration,
	bool *continueLoglikelihoodIteration,
	size_t const THREAD) {
	// Now indexing for i ranges [0, nDWIs-1] and j ranges [1, nParams] since first nParams is the theta(1)
	continueThetaIteration[THREAD] = true;
	nIterTheta[THREAD] = 0;
	loglikelihood_old[THREAD] = loglikelihood[THREAD];
	while (continueThetaIteration[THREAD]) {
		nIterTheta[THREAD]++;
		calculateFisherInformation(fisherInformation, fisherInformation_sym, Z, score, DeltaTheta, expo, EN, expScaling, nDWIs, nParams, nDeltaParams, THREAD);
		
		// Optimize loglikelihood
		iterateLoglikelihood(indx, score, vv, DeltaTheta, Z, expo, theta, loglikelihood, loglikelihood_old, new_theta, regulatorLambda, fisherInformation, fisherInformation_sym, ZTheta, c, scaling, expScaling, EN, nDWIs, nParams, nDeltaParams, regulatorLambda0, regulatorRescaling, nIterLoglikelihood, iterLimitLoglikelihood, toleranceLoglikelihood, continueLoglikelihoodIteration, THREAD);
		//goto THE_END_THETA;
		DeltaThetaScore[THREAD] = 0.0;
		for (int i = 0; i < nDeltaParams; i++) {
			DeltaThetaScore[THREAD] += DeltaTheta[THREAD * nDeltaParams + i]
				* score[THREAD * nDeltaParams + i];
		}
		
		// Check if new loglikelihood is NaN, if not 
		// update theta(2:end) and loglikelihood_old
		if (loglikelihood[THREAD] != loglikelihood[THREAD]) {
		// NaN, don't update variables
			continueThetaIteration[THREAD] = (nIterTheta[THREAD] < iterLimitTheta);
		} else {
			for (int i = 1; i < nParams; i++) {
				theta[THREAD * nParams + i] = new_theta[THREAD * nDeltaParams + i - 1];
			}
			loglikelihood_old[THREAD] = loglikelihood[THREAD];
		
		continueThetaIteration[THREAD] =
			(((DeltaThetaScore[THREAD] > toleranceTheta)
				||
				((loglikelihood[THREAD] - loglikelihood_old[THREAD]) > toleranceLoglikelihood))
				&&
				(nIterTheta[THREAD] < iterLimitTheta));
		}
	}
	//THE_END_THETA:
}

__device__ void calculateNorms(
	double *norm1,
	double *norm2,
	double *theta,
	double *theta_old,
	const unsigned int nParams,
	size_t const THREAD) {
	
	norm1[THREAD] = 0.0;
	norm2[THREAD] = 0.0;
	for (int i = 0; i < nParams; i++) {
		norm1[THREAD] += theta_old[THREAD * nParams + i] * theta_old[THREAD * nParams + i];
		norm2[THREAD] += (theta[THREAD * nParams + i] - theta_old[THREAD * nParams + i])*
			(theta[THREAD * nParams + i] - theta_old[THREAD * nParams + i]);
	}
	norm1[THREAD] = sqrt(norm1[THREAD]);
	norm2[THREAD] = sqrt(norm2[THREAD]);
}

__global__ void RicianMLE(
	double *theta,
	double *SigmaSQ,
	double *Z,
	double *fisherInformation,
	double *fisherInformation_sym,
	double *score,
	double *DeltaTheta,
	double *new_theta,
	double *vv,
	int *indx,
	double *theta_old,
	double *Y,
	double *expZTheta,
	double *ZTheta,
	double *twotau,
	double *expo,
	double *EN,
	double *b,
	double *a,
	double *c,
	double *sumYSQ,
	double *theta1_old,
	double *SigmaSQ0,
	double *SigmaSQ_old,
	double *tmpdouble,
	double *scaling,
	double *expScaling,
	double *loglikelihood,
	double *loglikelihood_old,
	double *regulatorLambda,
	double *DeltaThetaScore,
	double *norm1,
	double *norm2,
	unsigned int *nIterSigmaSQ,
	unsigned int *nIterVoxel,
	unsigned int *nIterS0,
	unsigned int *nIterTheta,
	unsigned int *nIterLoglikelihood,
	bool *continueSigmaSQIteration,
	bool *continueVoxelIteration,
	bool *continueS0Iteration,
	bool *continueThetaIteration,
	bool *continueLoglikelihoodIteration,
	bool *anyEN,
	const double toleranceSigmaSQ,
	const double toleranceS0,
	const double toleranceTheta,
	const double toleranceLoglikelihood,
	const unsigned int iterLimitSigmaSQ,
	const unsigned int iterLimitVoxel,
	const unsigned int iterLimitS0,
	const unsigned int iterLimitTheta,
	const unsigned int iterLimitLoglikelihood,
	const double regulatorLambda0,
	const double regulatorRescaling,
	const unsigned int nDWIs,
	const unsigned int nParams,
	const unsigned int nDeltaParams,
	const unsigned int nVoxels) {
	
	// Initial, work out which THREAD i.e. voxel we are computing
	size_t const THREAD = calculateGlobalIndex();
	if (THREAD >= nVoxels) {
		return;
	}
	
	// First, optimize Rician loglikelihood w.r.t. SigmaSQ
	calculateExpZTheta( expZTheta, theta, Z, nParams, nDWIs, THREAD);
	calculateAB_1(a, b, Y, expZTheta, sumYSQ, nDWIs, THREAD);
	iterateSigmaSQ(SigmaSQ, SigmaSQ0, tmpdouble, a, b, twotau, nIterSigmaSQ, iterLimitSigmaSQ, toleranceSigmaSQ, nDWIs, continueSigmaSQIteration, THREAD);
	
	// Start voxel-wise optimization
	continueVoxelIteration[THREAD] = true;
	while (continueVoxelIteration[THREAD]) {
		nIterVoxel[THREAD]++;
		// Save initial theta and SigmaSQ to be used later to test if voxel optimization continues
		SigmaSQ_old[THREAD] = SigmaSQ[THREAD];
		for (int i = 0; i < nParams; i++) {
			theta_old[THREAD * nParams + i] = theta[THREAD * nParams + i];
		}
		// Second, optimize w.r.t. S0 i.e. theta(1) with fixed theta(2:end) and SigmaSQ				
		// calcuateAB_2 updates a,b, expZTheta, and twotau variables
		calculateAB_2(a, b, Y, Z, theta, SigmaSQ, expZTheta, twotau, nDWIs, nParams, THREAD);
		// iterateS0 updates theta(1) and twotau variables
		iterateS0(theta, theta1_old, SigmaSQ, a, b, twotau, nIterS0, iterLimitS0, toleranceS0, nDWIs, nParams, continueS0Iteration, THREAD);
		
		// Third, optimize w.r.t. theta(2:end) with fixed theta(1) and SigmaSQ
		// calculateEN updates conditional expectation EN and checks if any(EN > 0) 
		calculateEN(EN, twotau, nDWIs, anyEN, THREAD);
		
		if (anyEN[THREAD]) {
			// There is information to estimate tensor(s)
			// calculateZTheta updates c and ZTheta variables
			calculateZTheta(c, ZTheta, theta, SigmaSQ, Z, nDWIs, nParams, THREAD);
			scaling[THREAD] = getMax(ZTheta, nDWIs, THREAD);
			expScaling[THREAD] = exp(scaling[THREAD]);			
			// calculateLoglikelihood updates loglikelihood and expo variables
			calculateLoglikelihood(loglikelihood, expo, ZTheta, scaling, expScaling, EN, nDWIs, THREAD);			
			initializeInformationMatrices(fisherInformation, fisherInformation_sym, nDeltaParams, THREAD);			
			iterateTheta(indx, vv, theta, ZTheta, c, fisherInformation, fisherInformation_sym, score, Z, EN, scaling, expScaling, expo, DeltaTheta, DeltaThetaScore, new_theta, loglikelihood, loglikelihood_old, regulatorLambda, regulatorLambda0, regulatorRescaling, nDWIs, nParams, nDeltaParams, nIterTheta, nIterLoglikelihood, iterLimitTheta, iterLimitLoglikelihood, toleranceTheta, toleranceLoglikelihood, continueThetaIteration, continueLoglikelihoodIteration, THREAD);					
			//goto THE_END;
		}
		else {
			// There is no information for estimations
			// Set theta(2:end) and information to zero
			for (int i = 1; i < nParams; i++) {
				theta[THREAD * nParams + i] = 0.0;
			}
			initializeInformationMatrices(fisherInformation, fisherInformation_sym, nDeltaParams, THREAD);
		}
		
		// Last, optimize w.r.t. SigmaSQ with fixed theta
		calculateExpZTheta(expZTheta, theta, Z, nParams, nDWIs, THREAD);
		calculateAB_1(a, b, Y, expZTheta, sumYSQ, nDWIs, THREAD);
		iterateSigmaSQ(SigmaSQ, SigmaSQ0, tmpdouble, a, b, twotau, nIterSigmaSQ, iterLimitSigmaSQ, toleranceSigmaSQ, nDWIs, continueSigmaSQIteration, THREAD);
		
		calculateNorms(norm1, norm2, theta, theta_old, nParams, THREAD);

		continueVoxelIteration[THREAD] =
			(((fabs((SigmaSQ[THREAD] - SigmaSQ_old[THREAD]) / SigmaSQ_old[THREAD]) > toleranceSigmaSQ)
				||
				((norm2[THREAD] / norm1[THREAD]) > toleranceTheta))
				&&
				(nIterVoxel[THREAD] < iterLimitVoxel));
	}
	//THE_END:
}