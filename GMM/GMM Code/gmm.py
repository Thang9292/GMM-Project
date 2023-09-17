import numpy as np
from tqdm import tqdm


SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

# Set False if the covariance matrix is a diagonal matrix
FULL_MATRIX = False

class GMM(object):
    def __init__(self, X, K, max_iters=100):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function 
    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error. 
        """

        #np.max takes in a single array and outputs the max element, the axis = 1 means that it will find the max
        #element in each row and return an array of size N containing those elements. Then if we want to do N x D - N x 1 that
        # way, the max array can be broadcasted downwards and affect every element
        stableLogit = logit - np.max(logit, axis = 1)[:, np.newaxis]
        #print(np.max(logit, axis = 1)[:, np.newaxis].shape)
        
        expLogit = np.exp(stableLogit)
        prob = expLogit / (np.sum(expLogit, axis = 1, keepdims=True))

        #print(expLogit.shape)
        return prob

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        #In this case, we have to add back the max, so keep track of it
        N = logit.shape[0]
        maxSum = np.max(logit, axis = 1)
        stableLogit = logit - np.max(logit, axis = 1)[:, np.newaxis]
        expLogit = np.exp(stableLogit)
        
        return (np.log(np.sum(expLogit, axis=1)) + maxSum).reshape((N, 1))
        

    # for undergraduate student
    def normalPDF(self, points, mu_i, sigma_i):
        """
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian

        """

        N = points.shape[0]
        D = points.shape[1]

        #np.diagonal() will return an array (D, ) containing the diagonal elements
        variance = np.diagonal(sigma_i)

        pdf = np.ones(N)
        #print(points.T[1].shape)

        for i in range(D):
            expPart = np.exp((-(( points.T[i] - mu_i[i]))**2) / (2*variance[i]))
            coefficientPart = np.sqrt(2 * np.pi * variance[i])
            
            #print("expPart has shape " + str(expPart.shape))
            pdf = pdf * (expPart / coefficientPart)
        
        #print(pdf)
        return pdf

    def _init_components(self, **kwargs):
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        """
        np.random.seed(5)

        K = self.K
        N = self.N
        D = self.D
        points = self.points
        max_iters = self.max_iters

        pi = np.ones(K)
        mu = np.zeros((K, D))
        sigma = np.zeros((K, D, D))
        
        pi = pi / K

        mu = points[np.random.randint(low = 0, high = N, size = K)]

        for i in range(K):
            sigma[i] = np.eye(D)

        return pi, mu, sigma


    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """

        if full_matrix is False:
            
            #Init some variables
            N = self.points.shape[0]
            K = mu.shape[0]
            ll = np.zeros((N, K))

            #Looping through K clusters
            for i in range(K):
                logPi = np.log(pi[i] + 1e-32)
                logGaussian = np.log(self.normalPDF(self.points, mu[i], sigma[i]) + 1e-32)
                #print(logGaussian.size)
                #print(logPi.size)
                #logGaussian is (N,), and logPi is a scalar so logPi + logGaussian is a (N, )

                #Fill in column i of ll with N elements by doing ll.T (N, K) -> (K, N)
                ll.T[i] = logPi + logGaussian

        return ll

    def _E_step(self, pi, mu, sigma, full_matrix = FULL_MATRIX , **kwargs):
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        """
        gamma_tau = np.zeros((self.N, self.K))
        if full_matrix is False:
            
            #ll is logPi + logGaussian, which when exponeniated will result in e^logPi + logGaussian or e^logPi * e^logGaussian = piGaussian
            #Softmax --> exp(logit) / sum(exp(logit)) which is exactly Tau
            loglikelyhood = self._ll_joint(pi, mu, sigma, FULL_MATRIX)
            gamma_tau = self.softmax(loglikelyhood)

        return gamma_tau

    def _M_step(self, gamma, full_matrix=FULL_MATRIX, **kwargs):
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        """
        if full_matrix is False:
            #Init some variables
            N = gamma.shape[0]
            K = gamma.shape[1]
            D = self.points.shape[1]
            N_k = gamma.sum(axis = 0)
            #print(N_k)
            #N_k has shape (K, )
            #We do axis = 0 so we sum over each row. So r1 + r2 + ... and get a single row/array.
            #This lets us get the sum of values in the K columns of gamma

            #Calculating mu
            #Requires matrix multiplication --> use @ symbol, Gamma.T is KxN. and self.points is NxD
            #gamma.T @ self.points is KxK. Then N_k is D, so we want to broad cast by doing a reshape to K x 1 so we get a
            #print( (gamma.T @ self.points).shape)
            mu = ((gamma.T @ self.points) / N_k[:, np.newaxis])

            #Calulating Sigma
            sigma = np.zeros((K, D, D))
            for i in range(K):
                #we do np.eye(D) so we only have the diagonal elements
                #gamma.T[i] is (N, ) and (self.points - mu[i]).T is D x N meaning we can broad cast it
                #(gamma.T[i] * (self.points - mu[i]).T @ (self.points - mu[i])) is DxD
                sigma[i] = np.eye(D) * (gamma.T[i] * (self.points - mu[i]).T @ (self.points - mu[i])) / gamma.T[i].sum(axis = 0)

            #Calculating pi
            pi = N_k / N

        return pi, mu, sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)