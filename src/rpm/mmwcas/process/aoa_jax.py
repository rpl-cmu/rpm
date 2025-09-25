"""Angle of Arrival processing using JAX."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, PyTree


class AoAJAX:
    """Angle of Arrival Point Cloud Processing Module using JAX.

    Args:
        gamma: Threshold factor for peak detection.
    """

    def __init__(
        self,
        gamma: float = 10 ** (0.2 / 10),
    ):
        self.gamma = gamma

    def peak_detect(
        self, signal: Complex[Array, "n"], gamma: float, sidelobe_level_db: float
    ):
        N = signal.shape[0]
        spectrum = jnp.abs(signal)

        state = {
            "i": 0,
            "curVal": 0,
            "iloc": 0,
            "extendLoc": 0,
            "initStage": True,
            "minVal": jnp.inf,
            "maxVal": -jnp.inf,
            "maxLoc": 0,
            "locateMax": False,
            "max_locs": jnp.zeros(N, dtype=bool),
            "spectrum": spectrum,
        }

        def update_maxloc(args: PyTree):
            args["max_locs"] = args["max_locs"].at[args["maxLoc"]].set(True)
            args["minVal"] = args["curVal"]
            args["locateMax"] = False
            return args

        def locate_next_peak(args: PyTree):
            args["locateMax"] = True
            args["maxVal"] = args["curVal"]

            def update_extend_loc(args: PyTree):
                args["extendLoc"] = args["i"]
                args["initStage"] = False
                return args

            args = jax.lax.cond(args["initStage"], update_extend_loc, lambda x: x, args)

            return args

        def update_t(args: PyTree):
            return jax.lax.cond(
                args["curVal"] < args["maxVal"] / gamma,
                update_maxloc,
                lambda x: x,
                args,
            )

        def update_f(args: PyTree):
            return jax.lax.cond(
                args["curVal"] > args["minVal"] * gamma,
                locate_next_peak,
                lambda x: x,
                args,
            )

        def update_max(args: PyTree):
            args["maxVal"] = args["curVal"]
            args["maxLoc"] = args["iloc"]
            return args

        def peak_fn(state: PyTree):
            state["iloc"] = state["i"] % N
            state["curVal"] = state["spectrum"][state["iloc"]]

            state = jax.lax.cond(
                state["curVal"] > state["maxVal"], update_max, lambda x: x, state
            )
            state["minVal"] = jnp.where(
                state["curVal"] < state["minVal"], state["curVal"], state["minVal"]
            )
            state = jax.lax.cond(state["locateMax"], update_t, update_f, state)
            state["i"] = state["i"] + 1
            return state

        def cond_fn(state: PyTree):
            return state["i"] < (N + state["extendLoc"] - 1)

        state = jax.lax.while_loop(cond_fn, peak_fn, state)
        sideLobeThresh = jnp.max(spectrum) * (10 ** (-sidelobe_level_db / 10))
        return jnp.logical_and(spectrum >= sideLobeThresh, state["max_locs"])

    def __call__(
        self,
        signal_azi: Complex[Array, "azi"],
        signal_angle: Complex[Array, "azi ele"],
    ):
        indx_azi = self.peak_detect(signal_azi, gamma=self.gamma, sidelobe_level_db=1)

        for i_azi in np.argwhere(indx_azi):
            indx_ele = self.peak_detect(
                signal_angle[np.squeeze(i_azi)], gamma=self.gamma, sidelobe_level_db=0
            )
            for i_ele in np.argwhere(indx_ele):
                print(i_azi, i_ele)

        return indx_azi
