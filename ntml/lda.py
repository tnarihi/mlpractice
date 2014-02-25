import numpy as np

__author__ = 'takuya'

class LatentDirichletAllocationGibbsSampler(object):
    def __init__(self, n_topic=10, alpha=0.1, beta=0.1, n_itr=100):
        self.n_topic = n_topic
        self.alpha_K = alpha if hasattr(alpha, '__iter__') else np.array([alpha] * n_topic)
        self.beta = beta
        self.n_itr = n_itr

    def _convert_to_sparse_format(self, w_DxV):
        """
        Convert doc-word matrix to sparse format
        """
        n_doc, n_vocab = w_DxV.shape
        n_word  = w_DxV.sum()
        doc_N = np.zeros(n_word)
        word_N = np.zeros(n_word)
        topic_N = np.zeros(n_word)
        i = 0
        for d in xrange(n_doc):
            for w in xrange(n_vocab):
                count = w_DxV[d, w]
                if count == 0: continue
                doc_N[i:i+count] = d
                word_N[i:i+count] = w
                topic_N[i:i+count] = np.random.randint(0, self.n_topic, count) # initialize topic assignments
                i += count
        return doc_N, word_N, topic_N

    def fit(self, w_DxV, monitoring=None):
        """
        Infer wth sparse format
        """
        n_doc, n_vocab = w_DxV.shape
        doc_N, word_N, topic_N = self._convert_to_sparse_format(w_DxV)

        # Initialize topic counter
        n_KxD = np.zeros((self.n_topic, n_doc), dtype=np.int32)
        n_KxV = np.zeros((self.n_topic, n_vocab), dtype=np.int32)
        for d, w, z in zip(doc_N, word_N, topic_N):
            n_KxD[z, d] += 1
            n_KxV[z, w] += 1
        n_K = n_KxD.sum(axis=1)
        self.n_KxD = n_KxD
        self.n_KxV = n_KxV
        self.n_K = n_K

        # Iteration for inference
        for itr in xrange(self.n_itr):
            print '#itr=%d' %(itr)
            # Gibbs sampling step
            for n, (d, w, z) in enumerate(zip(doc_N, word_N, topic_N)):
                n_KxD[z, d] -= 1
                n_KxV[z, w] -= 1
                n_K[z] -= 1
                p_K = (n_KxD[:, d] + self.alpha_K) * (n_KxV[:, w] + self.beta) / (n_K + self.beta * n_vocab)
                z_new = np.random.multinomial(n=1, pvals=p_K/p_K.sum()).argmax()
                n_KxD[z_new, d] += 1
                n_KxV[z_new, w] += 1
                n_K[z_new] += 1
                topic_N[n] = z_new

            # Call monitoring method
            if monitoring is not None:
                monitoring(self, itr)

    def get_word_topic_distribution(self):
        """
        phi_KxV: |K| x |V| word topic distribution
        """
        return (self.n_KxV + self.beta) / (self.n_K + self.beta * self.n_KxV.shape[0])[:, np.newaxis]

    def get_document_topic_distribution(self):
        """
        theta_KxD: |K| x |D| document topic distribution
        """
        return (self.n_KxD + self.alpha_K[:, np.newaxis]) / (self.n_KxD.sum(axis=0) + self.alpha_K.sum())

    def infer_unseen_document(self, w_DxV):
        """
        """
        n_doc, n_vocab = w_DxV.shape
        doc_N, word_N, topic_N = self._convert_to_sparse_format(w_DxV)
        n_KxD = np.zeros((self.n_topic, n_doc), dtype=np.int32)
        n_KxV = np.zeros((self.n_topic, n_vocab), dtype=np.int32)
        for d, w, z in zip(doc_N, word_N, topic_N):
            n_KxD[z, d] += 1
            n_KxV[z, w] += 1
        n_K = n_KxD.sum(axis=1)

        # Iteration for inference
        for itr in xrange(self.n_itr):
            print '#itr=%d' %(itr)
            # Gibbs sampling step
            for n, (d, w, z) in enumerate(zip(doc_N, word_N, topic_N)):
                n_KxD[z, d] -= 1
                n_KxV[z, w] -= 1
                n_K[z] -= 1
                p_K = (n_KxD[:, d] + self.alpha_K) * (self.n_KxV[:, w] + n_KxV[:, w] + self.beta) \
                      / (self.n_K + n_K + self.beta * n_vocab)
                z_new = np.random.multinomial(n=1, pvals=p_K/p_K.sum()).argmax()
                n_KxD[z_new, d] += 1
                n_KxV[z_new, w] += 1
                n_K[z_new] += 1
                topic_N[n] = z_new
        return (n_KxD + self.alpha_K[:, np.newaxis]) / (n_KxD.sum(axis=0) + self.alpha_K.sum())
