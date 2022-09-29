import time
import numpy as np
import particles

from particles import distributions as dists
from particles import state_space_models as ssm
from particles import mcmc
from particles import smc_samplers as ssp

from DataReaderWriter import DataReaderWriter
from Plotter import Plotter
from TwoLegModelSMC import TwoLegModel
from CustomMCMC import TruncatedPMMH


def set_prior(add_legs, add_imus, add_alphas, set_theta0):
    prior_dict = {}
    theta0 = None
    if add_legs:
        prior_legs = {
            "femur_left": dists.Uniform(0.3, 0.7),
            "fibula_left": dists.Uniform(0.4, 0.8),
            "femur_right": dists.Uniform(0.3, 0.7),
            "fibula_right": dists.Uniform(0.4, 0.8),
        }
        prior_dict.update(prior_legs)
        if set_theta0:
            theta0 = np.array([(0.5, 0.5, 0.5, 0.5)],
                              dtype=[("femur_left", "float64"), ("fibula_left", "float64"),
                                     ("femur_right", "float64"), ("fibula_right", "float64")])
    if add_imus:
        prior_imus = {"pos_imu0": dists.Uniform(a=0.0, b=0.5), "pos_imu2": dists.Uniform(a=0.0, b=0.5)}
        prior_dict.update(prior_imus)
        if set_theta0:
            theta0 = np.array([(0.25, 0.25)], dtype=[("pos_imu0", "float64"), ("pos_imu2", "float64")])
    if add_alphas:
        prior_alphas = {"alpha0": dists.TruncNormal(mu=0.0, sigma=1.0, a=-2.0, b=2.0),
                        "alpha2": dists.TruncNormal(mu=0.0, sigma=1.0, a=-2.0, b=2.0)}
        prior_dict.update(prior_alphas)
        if set_theta0:
            theta0 = np.array([(0.0, 0.0)], dtype=[("alpha0", "float64"), ("alpha2", "float64")])
    return theta0, prior_dict, dists.StructDist(prior_dict)


def learn_model_parameters(theta0, prior_dict, structured_prior, learning_alg, Nx, N, t_start, niter, true_states, data,
                           dt, show_fig, export_name=None):
    if learning_alg == "pmmh":
        alg = mcmc.PMMH(
            ssm_cls=TwoLegModel,
            prior=structured_prior,
            fk_cls=ssm.GuidedPF,
            smc_options={"ESSrmin": 0.5},
            data=data,
            Nx=Nx,
            theta0=theta0,
            niter=niter,
            verbose=niter,
            adaptive=True,
            scale=1.0,
        )
    elif learning_alg == "cpmmh":
        alg = TruncatedPMMH(
            ssm_cls=TwoLegModel,
            prior=structured_prior,
            fk_cls=ssm.GuidedPF,
            smc_options={"ESSrmin": 0.5},
            data=data,
            Nx=Nx,
            theta0=theta0,
            niter=niter,
            verbose=niter,
            adaptive=True,
            scale=1.0,
            t_trunc=t_start,
        )
    elif learning_alg == "smc2":
        fk_smc2 = ssp.SMC2(
            ssm_cls=TwoLegModel,
            prior=structured_prior,
            fk_cls=ssm.GuidedPF,
            data=data,
            init_Nx=Nx,
            ar_to_increase_Nx=-1.0,
            smc_options={"verbose": True},
        )
        alg = particles.SMC(fk=fk_smc2, N=N)
    else:
        raise ValueError("learning_alg has to be one of 'pmmh', 'smc2'; got {} instead.".format(learning_alg))
    start_user, start_process = time.time(), time.process_time()
    alg.run()
    end_user, end_process = time.time(), time.process_time()
    s_user = end_user - start_user
    s_process = end_process - start_process
    print(
        "Time user {:.0f}h {:.0f}min; time processor {:.0f}h {:.0f}min".format(s_user // 3600,
                                                                               s_user % 3600 // 60,
                                                                               s_process // 3600,
                                                                               s_process % 3600 // 60))
    data_writer = DataReaderWriter()
    data_writer.export_parameters(alg, prior_dict, export_name)
    plotter = Plotter(np.array(true_states), np.array(data), dt, export_name, show_fig)
    plotter.plot_learned_parameters(alg, learning_alg, prior_dict)

    return None


if __name__ == "__main__":
    # ---------------------------- data ----------------------------
    generation_type = "Missingdata005"  # name of data subfolder
    nb_timesteps = 1000
    dim_obs = 20  # 20 or 36
    data_reader = DataReaderWriter()
    x, y = data_reader.get_data_as_lists(generation_type, nb_timesteps, dim_obs)
    dt = 0.01

    # ---------------------------- parameter learning ----------------------------
    add_legs = False
    add_imu = False
    add_alphas = True
    set_theta0 = True  # can only be used if not more than one parameter is chosen
    theta0, prior_dict, prior = set_prior(add_legs, add_imu, add_alphas, set_theta0)
    Nx = 1000
    N = 20  # only used for smc2
    t_trunc = 500
    niter = 100
    learning_alg = "cpmmh"  # choose sampler: cpmmh, pmmh, smc2
    show_fig = True
    prior_str = "_".join(prior_dict.keys())
    export_name = "Learning{}_data{}_steps{}_Nx{}_niter{}_ttrunc{}_prior{}".format(
        learning_alg, generation_type, nb_timesteps, Nx, niter, t_trunc, prior_str
    )
    learn_model_parameters(
        theta0=theta0,
        prior_dict=prior_dict,
        structured_prior=prior,
        learning_alg=learning_alg,
        Nx=Nx,
        N=N,
        t_start=t_trunc,
        niter=niter,
        true_states=x,
        data=y,
        dt=dt,
        show_fig=show_fig,
        export_name=export_name,
    )
