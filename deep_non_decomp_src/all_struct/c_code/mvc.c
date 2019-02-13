#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define MISCLASSFN 1
#define PREC 2
#define REC 3
#define HMEAN 4
#define QMEAN 5
#define FONE 6
#define KLDNORM 7
#define BER 8
#define NAS 9
#define NSS 10
#define MTPRTNR 11
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
struct valLocPair{
	double val;
	int loc;
};

typedef struct valLocPair vlPair;

/* Sort in ascending order */
int compVLPair(const void *a, const void *b){
	if(((vlPair*)a)->val > ((vlPair*)b)->val)
		return 1;
	else
		return -1;
}

/*
Regularize probability values so that KLD stays below log(1/eps)

eps: in the range [0,1]
p: probability value to be regularized
*/
double regularizeProbValue(double p, double eps)
{
  double ps = (p + eps)/(1.0 + 2.0*eps);
  return ps;
}

/*
Get the performance according to different measures

Note: a,b,c,d are numbers/counts not rates
Note: we need loss not reward versions of performance measures

a: number of true positives
b: number of false positives
c: number of false negatives
d: number of true negatives
lossFunction: loss function being used
*/
double getLossVal(int a, int b, int c, int d, int lossFunction){
	double loss;
	double tpr, tnr, prec, rec;
	double p, phat, preg, phatreg, eps, kld;
	tpr = ((double)a)/((double)(a+c));
	tnr = ((double)d)/((double)(b+d));
	
	if((a+b) == 0)
		prec = 0.0;
	else
		prec = ((double)a)/(double)(a+b);
	
	if((a+c) == 0)
		rec = 0.0;
	else
		rec = ((double)a)/((double)(a+c));
	
	eps = 1.0/(2.0*(double)(a+b+c+d));
	p = ((double)(a+c))/((double)(a+c+b+d));
	phat = ((double)(a+b))/((double)(a+c+b+d));
	preg = regularizeProbValue(p,eps);
	phatreg = regularizeProbValue(phat,eps);
	
  //printf("INSIDE%d\n", lossFunction);
	if(lossFunction == MISCLASSFN)
		loss = ((double)(b+c))/((double)(a+c+b+d));
	else if (lossFunction == PREC)
		loss = 1 - prec;
	else if (lossFunction == REC)
		loss = 1 - rec;
	else if (lossFunction == HMEAN)
		loss = 1 - 2*tpr*tnr/(tpr + tnr);
	else if (lossFunction == QMEAN)
		loss = sqrt((pow(1-tpr,2.0) + pow(1-tnr,2.0))/2);
	else if (lossFunction == FONE)
      loss = 1.0 - (2.0*prec*rec/(prec+rec));
	else if (lossFunction == KLDNORM)
		loss = (preg*log(preg/phatreg) + (1-preg)*log((1-preg)/(1-phatreg)))/log(1/eps);
	else if (lossFunction == BER)
		loss = 1 - (tpr+tnr)/2;
	else if (lossFunction == NAS)
		loss = abs((double)(b-c))/((double)(a+c+b+d));
	else if (lossFunction == NSS)
		loss = pow((double)(b-c)/((double)(a+c+b+d)),2.0);
  else if (lossFunction == MTPRTNR)
    loss = 1 - min(tpr, tnr);
  return loss;
}


/*
Get the most violated constraint in the structural SVM corresponding to different measures

y: stream of true labels in {-1,1}
scores: scores given to these data points by the current model, e.g. the score to i-th data point is <w,x_i> for a linear model w
lossFunction: loss function being used
tot: number of feature vectors being supplied (must be equal to number of scores being supplied)
p: number of positive points
n: number of negative points
*/
double* getMostViolatedConstraint(double *y, double *scores,
                                  int lossFunction, int tot, int p, int n, double *yhat){
  //double* yhat = (double*)malloc(tot*sizeof(double));
  int dataCounter;
	vlPair* scoreLocPairs = (vlPair*)malloc(tot*sizeof(vlPair));
	double* cumSumPos = (double*)malloc((p+1)*sizeof(double));
	double* cumSumNeg = (double*)malloc((n+1)*sizeof(double));
	
	for(dataCounter=0; dataCounter<tot; dataCounter++){
		(scoreLocPairs + dataCounter)->loc = dataCounter;
		(scoreLocPairs + dataCounter)->val = *(scores + dataCounter);
	}
	
	/* Sort the points in ascending order of their scores */
  qsort(scoreLocPairs, tot, sizeof(vlPair), compVLPair);
	int posCounter = 0;
	int negCounter = 0;        
	double curPosSum = 0;
	double curNegSum = 0;
	double totPosSum, totNegSum;
	
	*cumSumPos = 0;
	*cumSumNeg = 0;
	
	/* cumSumPos[i] = Sum of scores of 'i' positives with highest scores */
	for(dataCounter = tot-1; dataCounter >= 0; dataCounter--){
		if(*(y + (scoreLocPairs + dataCounter)->loc) == 1){
			curPosSum += (scoreLocPairs + dataCounter)->val;
			posCounter++;
			*(cumSumPos + posCounter) = curPosSum;                
		}
	}
	totPosSum = curPosSum;
  /* cumSumNeg[i] = Sum of scores of 'i' negatives with lowest scores */
	for(dataCounter = 0; dataCounter < tot; dataCounter++){
		if(*(y + (scoreLocPairs + dataCounter)->loc) == -1){
			curNegSum += (scoreLocPairs + dataCounter)->val;
			negCounter++;
			*(cumSumNeg + negCounter) = curNegSum;
		}
	}
	totNegSum = curNegSum;
	
  double loss, JFMScore, currSurrogateLoss, maxSurrogateLoss;
	int numPosOpt = -1;
	int numNegOpt = -1;
	
	int numCandidatePos, numCandidateNeg, a, b, c, d; 
	
	/* Iterate over all possible TP and TN values */
	
	for(numCandidatePos = 0; numCandidatePos <= p; numCandidatePos++){                
		for(numCandidateNeg = 0; numCandidateNeg <= n; numCandidateNeg++){
			// Figure out the confusion matrix for this setting
			a = numCandidatePos;
			b = n - numCandidateNeg;
			c = p - numCandidatePos;
			d = numCandidateNeg;
			
			// Figure out the loss value for this confusion matrix
			loss = getLossVal(a,b,c,d,lossFunction);
			// Figure out the joint feature map scores
			JFMScore = *(cumSumPos + numCandidatePos) - (totPosSum - *(cumSumPos + numCandidatePos)) 
							+ (totNegSum - *(cumSumNeg + numCandidateNeg)) - *(cumSumNeg + numCandidateNeg);
							  
			// Surrogate loss value for this confusion matrix
			currSurrogateLoss = loss + JFMScore/((double)tot);

			// Does this confusion matrix yeild a higher surrogate loss value?
			if(currSurrogateLoss > maxSurrogateLoss || numPosOpt == -1){
				maxSurrogateLoss = currSurrogateLoss;
				numPosOpt = numCandidatePos;
				numNegOpt = numCandidateNeg;
			}                
		}
	}
	
	posCounter = 0;	
	negCounter = 0;
	int currLabel;
	
	// Find the labelling corresponding to the most violated constraint
	for(dataCounter = tot-1; dataCounter >= 0; dataCounter--){
    currLabel = *(y + (scoreLocPairs + dataCounter)->loc);
		
		// Assign numPosOpt top ranked positive data points a positive label, rest a negative label
		// Assign numNegOpt bottom ranked negative data points a negative label, rest a positive label
    if(currLabel == 1 && posCounter < numPosOpt){
			*(yhat + (scoreLocPairs + dataCounter)->loc) = 1.0;
			posCounter++;
    }else if(currLabel == 1 && posCounter >= numPosOpt){
			*(yhat + (scoreLocPairs + dataCounter)->loc) = -1.0;
      posCounter++;
		}else if(currLabel == -1 && negCounter < n - numNegOpt){
			*(yhat + (scoreLocPairs + dataCounter)->loc) = 1.0;
      negCounter++;
		}else if(currLabel == -1 && negCounter >= n - numNegOpt){
			*(yhat + (scoreLocPairs + dataCounter)->loc) = -1.0;
      negCounter++;
		}   
	}
	
}


/*
Get the subgradient of the structural SVM corresponding to different measures

X: stream of feature vectors for data points
y: stream of true labels in {-1,1}
scores: scores given to these data points by the current model, e.g. the score to i-th data point is <w,x_i> for a linear model w
lossFunction: loss function being used
tot: number of feature vectors being supplied (must be equal to number of labels being supplied)
d: dimensionality of the feature vectors
*/
double* getSubGradient(double* X, double* y, double* scores, int lossFunction, int tot, int d){
	
	double* grad = (double*)malloc(d*sizeof(double));
	double* alpha = (double*)malloc(tot*sizeof(double));
	int dataCounter;
	int p = 0;
  int n = 0;
	
	for(dataCounter = 0; dataCounter < tot; dataCounter++){
		if(*(y + dataCounter) > 0)
			p++;
		else
			n++;
	}
	double* yhat = (double*)malloc(tot*sizeof(double));
	getMostViolatedConstraint(y, scores, lossFunction, tot, p, n, yhat);
	
	p = 0;
	n = 0;
	
	// The gradient will be sum_i alpha_i x_i where x_i is the feature vector of the i-th data point
	// alpha_i values are calculated below
	for(dataCounter = 0; dataCounter < tot; dataCounter++){
		*(alpha + dataCounter) = *(yhat + dataCounter) - *(y + dataCounter);
	}
	
	for(int dCounter = 0; dCounter < d; dCounter++){
		*(grad + dCounter) = 0;
	}
	
	for(int dataCounter = 0; dataCounter < tot; dataCounter++){
		for(int dCounter = 0; dCounter < d; dCounter++){
			*(grad + dCounter) += *(X + d*dataCounter + dCounter)**(alpha + dataCounter);
		}
	}
	
	return grad;
}

int main()
{
  return 1;
}
