import pytest
import numpy as np
from alpa.adaptdl.pollux_agent import pollux_agent

@pytest.fixture
def setup_agent():
    pollux_agent.bs_dp = {10: 0.5, 20: 0.3}  # Example batch sizes and DynP costs
    pollux_agent.total_batch_size = 10
    return pollux_agent

def test_predict_dynp_cost_with_adequate_data(setup_agent):
    batch_sizes = np.array([[15], [25]])
    predictions = setup_agent.predict_dynp_cost(batch_sizes)
    assert predictions.shape == (2, 1)
    
def test_predict_dynp_cost_with_insufficient_data(setup_agent):
    setup_agent.bs_dp = {10: 0.5}  # Only one data point
    batch_sizes = np.array([[15]])
    with pytest.raises(AssertionError):
        setup_agent.predict_dynp_cost(batch_sizes)
        
def test_predict_dynp_cost_with_large_range(setup_agent):
    setup_agent.bs_dp = {i: 1.0 / i for i in range(10, 100, 10)}  # Larger range of data
    batch_sizes = np.array([[15], [50], [95]])
    predictions = setup_agent.predict_dynp_cost(batch_sizes)
    assert predictions.shape == (3, 1)

def test_predict_dynp_cost_for_validity(setup_agent):
    batch_sizes = np.array([[10], [20]])
    expected = np.array([[0.5], [0.3]])
    predictions = setup_agent.predict_dynp_cost(batch_sizes)
    np.testing.assert_array_almost_equal(predictions, expected, decimal=2)
    
# def test_predict_dynp_cost_with_non_integer(setup_agent):
#     batch_sizes = np.array([[15.5], [25.3]])
#     # with pytest.raises(ValueError):
#     print(setup_agent.predict_dynp_cost([batch_sizes]))
        
# def test_predict_dynp_cost_with_negative_values(setup_agent):
#     batch_sizes = np.array([[-10], [-20]])
#     # with pytest.raises(ValueError):
#     setup_agent.predict_dynp_cost(batch_sizes)