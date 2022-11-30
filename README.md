# MLE_for_dMRI
Maximum likelihood estimator for diffusion-weighted MRI.

References to cite if used:
1) Gasbarra, Dario, Jia Liu, and Juha Railavo. "Data augmentation in rician noise model and bayesian diffusion tensor imaging." arXiv preprint arXiv:1403.5065 (2014).
2) Liu, Jia, Dario Gasbarra, and Juha Railavo. "Fast estimation of diffusion tensors under Rician noise by the EM algorithm." journal of neuroscience methods 257 (2016): 147-158.
3) Sairanen, Viljami, Liu, J, Gasbarra, D. GPU Accelerated Maximum Likelihood Estimation of Diffusion and Kurtosis Tensors with the Rician Noise Model, International Society of Magnetic Resonance in Medicine (ISMRM), Paris, France, 2018.

A simple example:

a = dMLE('dwi.nii.gz', 'KT', 'mask.nii.gz', 'bvals', 'bvecs');

a = a.dMLE_fit(4,128); % M=4 tells into how many parts dataset is splitted, n_threads=128

a.dMLE_save('output_prefix');
