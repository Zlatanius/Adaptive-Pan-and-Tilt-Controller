"""
Nengo-based PID-style controller for a pan-tilt unit (PTU).

Sign convention: positive pan/tilt error corresponds to coordinates
that are to the left of, and above, the centre of the frame.
"""

import nengo
import numpy as np


class NengoController:
    """
    Drives a pan-tilt unit toward a target position using a Nengo
    spiking-neural-network implementation of proportional control.

    The controller encodes the proportional error

        k * (current_position - target_position)

    for both the pan and tilt axes in separate neural ensembles, and
    reads the (negated) decoded value back out as the control signal.
    """

    def __init__(
        self,
        k_pan=np.array([1.0, 0.0, 0.0]),
        k_tilt=np.array([1.0, 0.0, 0.0]),
        target_pan=64,
        target_tilt=64,
        depth=0.0,
        simulator_dt=0.001,
        n_neurons=100,
        ensemble_radius=50,
        probe_tau = 0.01,
    ):
        self.k_pan = k_pan
        self.k_tilt = k_tilt
        self.target_pan = target_pan
        self.target_tilt = target_tilt
        self.depth = depth
        self.simulator_dt = simulator_dt
        self.n_neurons = n_neurons
        self.ensemble_radius = ensemble_radius
        self.probe_tau = probe_tau

        # Initialize the current pan/tilt to the target so the
        # controller doesn't produce a large swing the moment it
        # starts running.
        self.pan = target_pan
        self.tilt = target_tilt

        self._build_network()

    def _build_network(self):
        """Construct and compile the Nengo model/simulator."""
        self.model = nengo.Network()

        with self.model:
            pan_inp = nengo.Node(lambda t: self.pan)
            tilt_inp = nengo.Node(lambda t: self.tilt)
            pan_target = nengo.Node(lambda t: self.target_pan)
            tilt_target = nengo.Node(lambda t: self.target_tilt)

            # Proportional error on pan: encodes k_pan * (pan - target_pan)
            pan_err_prop = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=1,
                radius=self.ensemble_radius,
                # intercepts=nengo.dists.Uniform(-1.0,1.0),
                # encoders = nengo.dists.Choice([[-1],[1]]),
            )
            nengo.Connection(pan_target, pan_err_prop, transform=-self.k_pan[0])
            nengo.Connection(pan_inp, pan_err_prop, transform=self.k_pan[0])

            # Proportional error on tilt: encodes k_tilt * (tilt - target_tilt)
            tilt_err_prop = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=1,
                radius=self.ensemble_radius,
                # intercepts=nengo.dists.Uniform(-1.0,1.0),
                # encoders = nengo.dists.Choice([[-1],[1]]), 
            )

            nengo.Connection(tilt_target, tilt_err_prop, transform=-self.k_tilt[0])
            nengo.Connection(tilt_inp, tilt_err_prop, transform=self.k_tilt[0])

            self.p_pan = nengo.Probe(
                pan_err_prop,
                synapse=self.probe_tau
            )
            self.p_tilt = nengo.Probe(
                tilt_err_prop,
                synapse=self.probe_tau
            )

        self.sim = nengo.Simulator(self.model, dt=self.simulator_dt)

    def __call__(self, salmax_coords, sim_run_time=0.5, avg_window=100):
        """
        Run the controller forward given the latest saliency-max
        coordinates and return the (tilt, pan) control signal.

        Parameters
        ----------
        salmax_coords : sequence of float
            (tilt, pan) coordinates of the saliency maximum.
        sim_run_time : float
            Simulation time to advance, in seconds.
        avg_window : int
            Number of trailing probe samples to average over.

        Returns
        -------
        (data_tilt, data_pan) : tuple of float
        """
        self.tilt, self.pan = salmax_coords[0], salmax_coords[1]

        # self.sim.reset()
        self.sim.run(sim_run_time, progress_bar=False)

        # Average the last `avg_window` samples of probe data.
        # Note: by default the Nengo simulator uses a 1 ms time step,
        # so probe data is indexed in milliseconds.
        # TODO: determine the optimal averaging window size.
        full_data_pan = self.sim.data[self.p_pan]
        full_data_tilt = self.sim.data[self.p_tilt]

        data_tilt = np.mean(self.sim.data[self.p_tilt][-avg_window:])
        data_pan = np.mean(self.sim.data[self.p_pan][-avg_window:])

        self.sim.clear_probes()
        return data_tilt, data_pan, full_data_pan, full_data_tilt
