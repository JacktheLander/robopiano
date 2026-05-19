# Etude

Etude is the low-level closed-loop trajectory-following controller for RoboPianist.
It consumes 46D playback trajectories generated upstream by Intermezzo or extracted
from RP1M, observes the live simulator state, and emits valid RoboPianist actions.

Etude does not generate playback trajectories. Its job is to answer:

```text
Given q_ref[t] at 200 Hz, can a controller drive RoboPianist closely enough to play the intended notes?
```

## Install

```bash
cd Etude
pip install -e .
```

Optional simulator dependencies:

```bash
pip install -e ".[full]"
```

## Smoke Test

```bash
python -c "import etude; print(etude.__version__)"
pytest
```

## Core Interfaces

Controllers implement:

```python
class TrajectoryFollower:
    def reset(self, q_ref, qdot_ref=None, metadata=None) -> None:
        ...

    def act(self, obs, t: int) -> np.ndarray:
        ...
```

The first useful baseline is `PDController`, followed by learned residual models
that correct the PD action while preserving a stabilizing tracking prior.
