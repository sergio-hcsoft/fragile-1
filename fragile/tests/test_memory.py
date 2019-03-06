import pytest
import numpy as np
from fragile.memory import Memory
from fragile.states import States


@pytest.fixture()
def memory():
    return Memory(10, 100, 1)


@pytest.fixture()
def mem_states():
    return States(n_walkers=50, observs=np.random.random((50, 32, 3)))


class TestMemory:
    def test_len_0(self, memory):
        assert len(memory) == 0

    def test_update_on_init(self, mem_states, memory):
        memory.update(mem_states)
        memory.update(observs=mem_states.observs)
        with pytest.raises(ValueError):
            memory.update(states=mem_states.observs)
        with pytest.raises(ValueError):
            memory.update()
        with pytest.raises(ValueError):

            class Dummy:
                @property
                def observs(self):
                    return None

            memory.update(Dummy())

    def test_update_values(self, mem_states, memory):
        memory.update(mem_states)
        assert len(memory) > 0

    def test_len(self, memory, mem_states):
        pass
