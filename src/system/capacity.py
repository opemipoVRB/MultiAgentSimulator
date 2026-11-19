# src/system/capacity.py
import psutil

from src.system.topology import physical_core_groups


class AgentCapacityError(RuntimeError):
    pass


class SystemCapacityChecker:
    """
    Estimates how many embodied SLM agents this machine can sustain.
    This is a hard feasibility gate, not a scheduler.
    """

    def __init__(
            self,
            ram_per_agent_gb: float = 1.2,
            cpu_cores_per_agent: int = 1,
            min_free_ram_gb: float = 1.0,
            verbose: bool = True,
            process_mode: bool = True,
            reserved_cores: int = 1,
    ):
        # process_mode: every drone is its own process on its own physical core, so cores limit the fleet; the world
        # process takes `reserved_cores`. Measured: a drone process is about 0.1 GB (a 12-drone run peaks near 1.2 GB
        # in total), so the default RAM figure is the embedded-model case; callers with a shared model server pass less.
        self.process_mode = process_mode
        self.reserved_cores = reserved_cores
        self.ram_per_agent_gb = ram_per_agent_gb
        self.cpu_cores_per_agent = cpu_cores_per_agent
        self.min_free_ram_gb = min_free_ram_gb
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[CAPACITY] {msg}")

    def max_supported_agents(self) -> int:
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        usable_ram = max(0.0, available_gb - self.min_free_ram_gb)
        max_by_ram = int(usable_ram // self.ram_per_agent_gb)

        physical_cores = len(physical_core_groups())          # physical cores this process may use
        logical_cores = psutil.cpu_count(logical=True) or physical_cores
        max_by_cpu = (max(0, physical_cores - self.reserved_cores) // self.cpu_cores_per_agent
                      if self.process_mode else 10 ** 6)

        # Logging
        self._log(f"Total RAM: {total_gb:.2f} GB")
        self._log(f"Available RAM: {available_gb:.2f} GB")
        self._log(f"Reserved free RAM buffer: {self.min_free_ram_gb:.2f} GB")
        self._log(f"Usable RAM for agents: {usable_ram:.2f} GB")
        self._log(f"RAM per agent: {self.ram_per_agent_gb:.2f} GB")
        self._log(f"Max agents by RAM: {max_by_ram}")

        self._log(f"Physical cores: {physical_cores}")
        self._log(f"Logical cores: {logical_cores}")
        self._log(f"Cores per agent: {self.cpu_cores_per_agent}")
        self._log(f"Max agents by CPU: {max_by_cpu}")

        final = max(1, min(max_by_ram, max_by_cpu))
        self._log(f"Final supported agent limit: {final}")

        return final

    def enforce(self, requested_agents: int):
        self._log(f"Requested agents: {requested_agents}")

        max_agents = self.max_supported_agents()

        if requested_agents > max_agents:
            self._log("Decision: REJECTED")
            raise AgentCapacityError(
                f"System can sustain at most {max_agents} embodied agents.\n"
                f"Requested: {requested_agents}\n\n"
                f"Reduce agent count or use a stronger machine."
            )

        self._log("Decision: ACCEPTED")
