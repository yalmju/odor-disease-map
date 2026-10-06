# Methods and interpretation

Research snapshot: 532 fixed coordinates. Historical projection settings recorded in the working project are L2-normalized OpenPOM 256D embeddings, cosine t-SNE, perplexity 30, seed 42, PCA initialization, auto learning rate, and 2,000 iterations. The present package reuses coordinates; it cannot independently reconstruct their upstream provenance, training split, checkpoint, or t-SNE result without the original pipeline. It does not infer clinical or perceptual distances from 2D geometry.

Odor fields sum isotropic Gaussian kernels with bandwidth 2.6 in plot units on a 540×420 grid. Each label is normalized to its own maximum, with opacity `0.97 * density**0.65`. Label counts differ, so darkness cannot compare prevalence or intensity across panels. Disease marks are exact linked points, without smoothing. Focal circles preserve gray for an unlinked record and pink for a linked record. Missing links are not evidence of absence.

Scientific limitations: associations do not establish causation, specificity, co-occurrence, diagnosis, or volatile abundance. Nearby t-SNE points do not prove perceptual equivalence. Known labels and predicted scores are separate evidence layers; the focal compound with a recorded fruity label has a lower predicted fruity score than its comparator. No SERS raw data or prospective spatial-map validation are included.

The full private input, reference figures, and model score CSV allow reproduction of the displayed figure stage. No medical recommendation is produced.
