import torch as th
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde
from scipy.stats import norm
from .base import BaseScheduler

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
        use_cosine_loss=False,
        use_lognorm=False,
        partitial_train=None,
        partial_ratio=1.0,
        shift_lg=False,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_cosine_loss = use_cosine_loss
        self.use_lognorm = use_lognorm
        self.partitial_train = partitial_train
        self.partial_ratio = partial_ratio
        self.shift_lg = shift_lg

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)


    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample_logit_normal(self, mu, sigma, size=1):
        # Generate samples from the normal distribution
        samples = norm.rvs(loc=mu, scale=sigma, size=size)

        # Transform samples to be in the range (0, 1) using the logistic function
        samples = 1 / (1 + np.exp(-samples))

        # Numpy to Tensor
        samples = th.tensor(samples, dtype=th.float32)

        return samples

    def sample_in_range(self, mu, sigma, target_size, range_min=0, range_max=0.5):
        samples = []
        while len(samples) < target_size:
            generated_samples = self.sample_logit_normal(mu, sigma, size=target_size)
            filtered_samples = generated_samples[(generated_samples >= range_min) & (generated_samples <= range_max)]
            samples.extend(filtered_samples)

        # If we have more than the target size, truncate the list
        samples = samples[:target_size]
        return th.tensor(samples)

    def sample(self, x_start, sp_timesteps=None, shifted_mu=0):
        """Sampling x0 & t based on shape of x_start (if needed)
          Args:
            x_start - data point; [batch, *dim]
        """

        x0 = th.randn_like(x_start)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if not self.use_lognorm:
            if self.partitial_train is not None and th.rand(1) < self.partial_ratio:
                t = th.rand((x_start.shape[0],)) * (self.partitial_train[1] - self.partitial_train[0]) + self.partitial_train[0]
            else:
                t = th.rand((x_start.shape[0],)) * (t1 - t0) + t0
        else:
            # random < partial_ratio, then sample from the partial range
            if not self.shift_lg:
                if self.partitial_train is not None and th.rand(1) < self.partial_ratio:
                    t = self.sample_in_range(0, 1, x_start.shape[0], range_min=self.partitial_train[0], range_max=self.partitial_train[1])
                else:
                    t = self.sample_logit_normal(0, 1, size=x_start.shape[0]) * (t1 - t0) + t0
            else:
                assert self.partitial_train is None, "Shifted lognormal distribution is not compatible with partial training"
                t = self.sample_logit_normal(shifted_mu, 1, size=x_start.shape[0]) * (t1 - t0) + t0

        # overwrite t if sp_timesteps is provided (for validation)
        if sp_timesteps is not None:
            # uniform sampling between self.sp_timesteps[0] and self.sp_timesteps[1]
            t = th.rand((x_start.shape[0],)) * (sp_timesteps[1] - sp_timesteps[0]) + sp_timesteps[0]

        t = t.to(x_start)
        return t, x0, x_start


    def training_losses(
        self,
        model,
        x_start,
        model_kwargs=None,
        t=None,
        noise=None,
        sp_timesteps=None,
        shifted_mu=0,
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x_start: datapoint
        - model_kwargs: additional arguments for the model
        - t: optional timestep tensor [batch]; sampled automatically if None
        - noise: optional noise tensor matching x_start shape; sampled automatically if None
        """
        if model_kwargs is None:
            model_kwargs = {}

        t_sampled, x0, x_start = self.sample(x_start, sp_timesteps, shifted_mu)
        if t is None:
            t = t_sampled
        if noise is not None:
            x0 = noise

        t, xt, ut = self.path_sampler.plan(t, x0, x_start)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C), f"model_output.shape: {model_output.shape}, xt.shape: {xt.shape}"

        terms = {}
        if self.model_type == ModelType.VELOCITY:
            terms['mse'] = mean_flat(((model_output - ut) ** 2))
            terms['loss'] = terms['mse']
            if self.use_cosine_loss:
                terms['cos_loss'] = mean_flat(1 - th.nn.functional.cosine_similarity(model_output, ut, dim=1))
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['mse'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['mse'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
            terms['loss'] = terms['mse']

        return terms


    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        model_kwargs=None,
        device=None,
        progress=False,
        sampler="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        """Generate samples by integrating the flow ODE from noise to data.

        Args:
        - model: the flow model (nn.Module or callable).
        - shape: output shape tuple (batch, *dims).
        - noise: optional initial noise; sampled from N(0,I) if None.
        - model_kwargs: additional arguments forwarded to the model.
        - device: target device; inferred from noise / model if None.
        - progress: unused, kept for interface parity with diffusion.
        - sampler: ODE solver type — ``"dopri5"`` (default, adaptive),
          ``"euler"`` or ``"heun"`` (fixed-step).
        - num_steps: integration steps (fixed solvers) or save-points (adaptive).
        - atol: absolute tolerance for adaptive solvers.
        - rtol: relative tolerance for adaptive solvers.

        Returns:
        - final sample tensor of the requested shape.
        """
        if device is None:
            if noise is not None:
                device = noise.device
            elif isinstance(model, th.nn.Module):
                device = next(model.parameters()).device
            else:
                device = th.device("cuda" if th.cuda.is_available() else "cpu")
        if noise is None:
            noise = th.randn(*shape, device=device)
        if model_kwargs is None:
            model_kwargs = {}

        _sampler = Sampler(self)
        sample_fn = _sampler.sample_ode(
            sampling_method=sampler,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        xs = sample_fn(noise, model, **model_kwargs)
        return xs[-1]

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn


    def get_score(
        self,
    ):
        """member function for obtaining score of
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)


        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        timestep_shift=0.0,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            timestep_shift=timestep_shift,
        )

        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):

        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn


# ─────────────────────────────────────────────────────────────────────────────
# Public wrapper: FlowMatchingScheduler
# ─────────────────────────────────────────────────────────────────────────────

class FlowMatchingScheduler(BaseScheduler):
    """Wraps Transport under the BaseScheduler interface.

    Provides training_losses and p_sample_loop with ODE/SDE samplers.
    All advanced Transport / Sampler methods (get_drift, get_score,
    sample_sde, sample_ode_likelihood, etc.) are accessible via
    attribute fall-through.
    """

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def training_losses(
        self,
        model,
        x_start,
        t=None,
        noise=None,
        model_kwargs=None,
    ):
        """Compute flow-matching training loss for one batch.
        Returns a dict with keys 'loss' and 'mse'.
        """
        terms = self._transport.training_losses(
            model, x_start,
            model_kwargs=model_kwargs,
            t=t,
            noise=noise,
        )
        self._collect_auxiliary_losses(model, terms)
        return terms

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        model_kwargs=None,
        device=None,
        progress=False,
        sampler="dopri5",
        **sampler_kwargs,
    ):
        """Generate samples by integrating the flow ODE from noise to data.

        Args:
            sampler: ODE solver — 'dopri5' (adaptive, default), 'euler', or 'heun'.
                     For SDE sampling use self.get_sampler().sample_sde(...) directly.
        """
        return self._transport.p_sample_loop(
            model, shape,
            noise=noise,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            sampler=sampler,
            **sampler_kwargs,
        )

    def __getattr__(self, name: str):
        """Fall through to Transport for advanced / low-level access."""
        return getattr(self._transport, name)

    def get_sampler(self) -> Sampler:
        """Return the underlying Sampler for advanced SDE/ODE access."""
        return Sampler(self._transport)


def create_transport(
    path_type='Linear',
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
    use_cosine_loss=None,
    use_lognorm=None,
    partitial_train=None,
    partial_ratio=1.0,
    shift_lg=False,
) -> FlowMatchingScheduler:
    """Factory for FlowMatchingScheduler.

    Args:
        path_type:     Interpolant path — 'Linear', 'GVP', or 'VP'.
        prediction:    Model output type — 'velocity' (default), 'score', or 'noise'.
        loss_weight:   Optional loss weighting — 'velocity', 'likelihood', or None.
        train_eps:     Epsilon to avoid instability at t=0 during training.
        sample_eps:    Epsilon to avoid instability at t=0 during sampling.
        use_cosine_loss: Add cosine similarity term to velocity MSE loss.
        use_lognorm:   Sample t from logit-normal distribution instead of uniform.
        partitial_train: (lo, hi) tuple to restrict training to a time sub-range.
        partial_ratio: Fraction of batches using the partial time range.
        shift_lg:      Shift the logit-normal distribution mean.

    Returns:
        FlowMatchingScheduler ready for training_losses / p_sample_loop.
    """
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    path_type_enum = path_choice[path_type]

    if path_type_enum in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    elif path_type_enum in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if train_eps is None else sample_eps
    else:
        train_eps = 0
        sample_eps = 0

    transport = Transport(
        model_type=model_type,
        path_type=path_type_enum,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
        use_cosine_loss=use_cosine_loss,
        use_lognorm=use_lognorm,
        partitial_train=partitial_train,
        partial_ratio=partial_ratio,
        shift_lg=shift_lg,
    )
    return FlowMatchingScheduler(transport)
