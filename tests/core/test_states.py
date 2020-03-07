import numpy
import pytest  # noqa: F401

from fragile.core.states import States, StatesEnv, StatesModel, StatesWalkers

state_classes = [States, StatesEnv, StatesModel, StatesWalkers]


class TestStates:
    @pytest.mark.parametrize("states_class", state_classes)
    def test_init_dict(self, states_class):
        state_dict = {"name_1": {"size": tuple([1]), "dtype": numpy.float32}}
        new_states = states_class(state_dict=state_dict, batch_size=2)
        assert new_states.n == 2

    @pytest.mark.parametrize("states_class", state_classes)
    def test_init_kwargs(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert new_states._n_walkers == 2
        assert name in new_states.keys()
        assert getattr(new_states, name) == name, type(new_states)

    @pytest.mark.parametrize("states_class", state_classes)
    def test_getitem(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert new_states[name] == name, type(new_states)

    @pytest.mark.parametrize("states_class", state_classes)
    def test_setitem(self, states_class):
        name_1 = "miau"
        val_1 = name_1
        name_2 = "elephant"
        val_2 = numpy.arange(10)
        new_states = states_class(batch_size=2)
        new_states[name_1] = val_1
        new_states[name_2] = val_2
        assert new_states[name_1] == val_1, type(new_states)
        assert (new_states[name_2] == val_2).all(), type(new_states)

    @pytest.mark.parametrize("states_class", state_classes)
    def test_repr(self, states_class):
        name = "miau"
        new_states = states_class(batch_size=2, miau=name)
        assert isinstance(new_states.__repr__(), str)

    @pytest.mark.parametrize("states_class", state_classes)
    def test_n(self, states_class):
        new_states = states_class(batch_size=2)
        assert new_states.n == new_states._n_walkers == 2

    @pytest.mark.parametrize("states_class", state_classes)
    def test_get(self, states_class):
        new_states = states_class(batch_size=2, test="test")
        assert new_states.get("test") == "test"
        assert new_states.get("AKSJDFKG") is None
        assert new_states.get("ASSKADFKA", 5) == 5

    @pytest.mark.parametrize("states_class", state_classes)
    def test_split_states(self, states_class):
        new_states = states_class(batch_size=10, test="test")
        for s in new_states.split_states():
            assert len(s) == 1
            assert s.test == "test"

    @pytest.mark.parametrize("states_class", state_classes)
    def test_get_params_dir(self, states_class):
        state_dict = {"name_1": {"size": tuple([1]), "dtype": numpy.float32}}
        new_states = states_class(state_dict=state_dict, batch_size=2)
        params_dict = new_states.get_params_dict()
        assert isinstance(params_dict, dict)
        for k, v in params_dict.items():
            assert isinstance(k, str)
            assert isinstance(v, dict)
            for ki, _ in v.items():
                assert isinstance(ki, str)

    @pytest.mark.parametrize("states_class", state_classes)
    def test_clone(self, states_class):
        batch_size = 10
        states = states_class(batch_size=batch_size)
        states.miau = numpy.arange(states.n)
        states.miau_2 = numpy.arange(states.n)

        will_clone = numpy.zeros(states.n, dtype=numpy.bool_)
        will_clone[3:6] = True
        compas_ix = numpy.array(numpy.arange(states.n)[::-1])

        states.clone(will_clone=will_clone, compas_ix=compas_ix)
        target_1 = numpy.arange(10)

        assert numpy.all(target_1 == states.miau), (target_1 - states.miau, states_class)
