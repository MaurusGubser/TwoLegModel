# TwoLegModelSMC
SSM implementation of a two leg motion model, using particles library for filtering, smoothing and parameter learning. The two leg model is implemented as a class, the SMC tasks like filtering, smoothing, parameter learning are done using the particles library.

The following files are implementation of classes:
- CustomDistributions.py sublasses ProbDist class of particles to implement custom distributions
- CustomMCMC.py subclasses PMMH class of particles to implement a custom PMMH sampler.
- DataReaderWriter.py is a class for reading and preparing data for the SMC sapmler of particles. Also, particles data can be exported.
- MechanicalModel.py implements the mechanical two leg model as a python class.
- Plotter.py implements a class for plotting the results of the SMC sampler, e.g. particle trajectories or likelihood plots.
- TwoLegEKFModel.py implements a general EKF filter for the two leg model, along with a short script for comparing EKF with SMC results.
- TwoLegSMCModel.py implements a SSM for the two leg model, i.e. probabilistic process and observation maps for forward computing.

The following files are scripts for executing SMC tasks for the two leg model:
- bayesian_parameter_learning.py can parameters like IMU-position, IMU-angle and others using PMMH or SMC2 methods.
- compare_model_with_truth.py compares the true observations with the observations based on the mechanical model and the true states.
- filtering_smoothing.py filters and smoothes state trajectories, given observations, using the SMC model.
- likelihood_analysing.py can be used to analyse the log likelihood of the data, given a model defined by certain parameters.
