import numpy as np
import pylab as pl
from matplotlib import gridspec
from ntml.lda import LatentDirichletAllocationGibbsSampler

__author__ = 'Narihira'

def visualize_squared_shape_word_dist(phi_KxV):
    n_topic, n_vocab = phi_KxV.shape
    r_vocab = int(np.sqrt(n_vocab))
    cols = int(np.ceil(np.sqrt(n_topic)))
    img = np.ones(((r_vocab+1) * cols - 1, (r_vocab + 1) * cols - 1)) * phi_KxV.mean()
    for r in xrange(cols):
        for c in xrange(cols):
            k = r * cols + c
            if k >= n_topic: continue
            img[r*(r_vocab+1):(r+1)*(r_vocab+1)-1, c*(r_vocab+1):(c+1)*(r_vocab+1)-1] = \
                phi_KxV[k, :].reshape(r_vocab, r_vocab)
    pl.imshow(img, interpolation='none', cmap=pl.gray())
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
        pl.subplot(gs[0]) ; pl.title('True')
        visualize_squared_shape_word_dist(datagen.phi_KxV)
        pl.subplot(gs[1]) ; pl.title('Inferred at %d'%itr)
        visualize_squared_shape_word_dist(phi_KxV)
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





