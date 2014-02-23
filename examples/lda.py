import numpy as np
import pylab as pl
from matplotlib import gridspec
__author__ = 'Narihira'

def visualize_squared_shape_word_dist(phi_KxV, subplot_spec):
    n_topic, n_vocab = phi_KxV.shape
    r_vocab = int(np.sqrt(n_vocab))
    cols = int(np.ceil(np.sqrt(n_topic)))
    gs = gridspec.GridSpecFromSubplotSpec(cols, cols, subplot_spec)
    for k in xrange(n_topic):
        pl.subplot(gs[k])
        pl.imshow(phi_KxV[k, :].reshape(r_vocab, r_vocab), interpolation='none', cmap=pl.gray())
        pl.axis('off')

class LdaDataGenerator(object):
    '''
    '''
    def __init__(self, n_topic=10, alpha=0.5):

        assert n_topic % 2 == 0 # n_topic must be even
        r_topic = n_topic / 2
        n_vocab = r_topic**2
        phi_KxV = np.zeros((n_topic, n_vocab)) # word distribution given topic

        # Create vertical word distributions of topics
        for z in xrange(r_topic):
            wdist = np.zeros((r_topic, r_topic))
            wdist[:,z] = 1.0
            wdist /= wdist.sum()
            phi_KxV[z, :] = wdist.flatten()

        # Create horizontal
        for z in xrange(r_topic):
            wdist = np.zeros((r_topic, r_topic))
            wdist[z,:] = 1.0
            wdist /= wdist.sum()
            phi_KxV[r_topic + z, :] = wdist.flatten()

        self.phi_KxV = phi_KxV
        self.n_topic = n_topic # shortcut
        self.n_vocab = n_vocab # shortcut
        self.alpha_K = np.array([alpha]*self.n_topic) # Dirichlet prior of topic distribution of document

    def generate_document(self, n_word=100):
        theta_K = np.random.dirichlet(alpha=self.alpha_K) # sample topic distribution of document
        z_K = np.random.multinomial(n=n_word, pvals=theta_K) # topic histogram
        w_V = np.zeros(self.n_vocab)
        for i, count in enumerate(z_K):
            w_V += np.random.multinomial(n=count, pvals=self.phi_KxV[i, :])
        return w_V, z_K


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

def main(n_doc=100):
    # Generate toy dataset
    datagen = LdaDataGenerator(n_topic=10, alpha=0.5)
    n_DxV, p_DxK = (), ()
    for d in xrange(n_doc):
        n_V, p_K = datagen.generate_document(n_word=100)
        n_DxV += n_V,
        p_DxK += p_K,
    n_DxV = np.array(n_DxV)
    p_DxK = np.array(p_DxK)

    # Init LDA learner
    lda = LatentDirichletAllocationGibbsSampler(n_topic=datagen.n_topic, alpha=1.0, beta=1.0, n_itr=100)

    # Define callback method
    pl.ion(); pl.figure('Word topic dist.')
    def lda_monitoring(lda, itr):
        if itr != 0 and (itr % 10 != 0) and itr != lda.n_itr - 1: return
        phi_KxV = lda.get_word_topic_distribution()
        gs = gridspec.GridSpec(1,2)
        visualize_squared_shape_word_dist(datagen.phi_KxV, gs[0])
        visualize_squared_shape_word_dist(phi_KxV, gs[1])
        pl.draw()
        pl.show()

    # Infer
    lda.fit(n_DxV, monitoring=lda_monitoring)

    # Show document topic distribution
    theta_KxD = lda.get_document_topic_distribution()
    # Find correspondence between true topics and inferred ones
    dist_KxK = (p_DxK.T**2).sum(axis=1)[:, np.newaxis] \
        - 2 * np.dot(p_DxK.T, theta_KxD.T) \
        + (theta_KxD.T**2).sum(axis=0)[np.newaxis]
    corr_K = dist_KxK.argmin(axis=1)
    pl.figure('Topic dist. of training documents')
    pl.subplot(211); pl.title('true p(z|d)')
    pl.imshow(p_DxK.T, interpolation='none', cmap=pl.gray())
    pl.subplot(212); pl.title('inferred p(z|theta_d)')
    pl.imshow(theta_KxD[corr_K, :], interpolation='none', cmap=pl.gray())
    pl.show()

    # Generate unseen documents
    n_us_DxV, p_us_DxK = (), ()
    for d in xrange(20):
        n_V, p_K = datagen.generate_document(n_word=100)
        n_us_DxV += n_V,
        p_us_DxK += p_K,
    n_us_DxV = np.array(n_us_DxV)
    p_us_DxK = np.array(p_us_DxK)

    # Infer unseen documents with trained model
    theta_us_KxD = lda.infer_unseen_document(n_us_DxV)

    # Show topic distributions of unseen documents
    pl.ioff(); pl.figure('Topic dist. of unseen documents')
    pl.subplot(211); pl.title('true p(z|d)')
    pl.imshow(p_us_DxK.T, interpolation='none', cmap=pl.gray())
    pl.subplot(212); pl.title('inferred p(z|theta_d)')
    pl.imshow(theta_us_KxD[corr_K], interpolation='none', cmap=pl.gray())
    pl.show()

if __name__ == '__main__':
    main()





