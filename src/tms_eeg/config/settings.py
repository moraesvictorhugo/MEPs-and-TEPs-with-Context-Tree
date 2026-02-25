from dataclasses import dataclass, field

@dataclass
class FilterConfig:
    l_freq: float = 1.0
    h_freq: float = 100.0
    notch_freq: float = 60.0

@dataclass
class EpochConfig:
    tmin: float = -0.5
    tmax: float = 0.5
    baseline: tuple = (-0.5, -0.01)

@dataclass
class TMSConfig:
    pulse_window: tuple = (-0.01, 0.01)  # seconds around TMS pulse
    interpolate: bool = True

# Overall project configuration
# Using field(default_factory=...) to ensure that each instance of ProjectConfig
# gets its own instance of the nested configs
@dataclass
class ProjectConfig:
    filter: FilterConfig = field(default_factory=FilterConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    tms: TMSConfig = field(default_factory=TMSConfig)
    subject_id: str = ""
    session: str = ""
