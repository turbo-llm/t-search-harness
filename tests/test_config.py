from retriever_agent import AgentConfig


def test_defaults():
    c = AgentConfig()
    assert c.temperature == 0.7
    assert c.top_p == 1.0
    assert c.max_rounds == 5
    assert c.max_tokens_per_turn == 16384
    assert c.budget_tokens == 32768
    assert c.hard_lock_ratio == 0.75
    assert c.model is None


def test_from_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model: my-model\nmax_rounds: 3\ntemperature: 0.5\n")
    c = AgentConfig.from_yaml(p)
    assert c.model == "my-model"
    assert c.max_rounds == 3
    assert c.temperature == 0.5
    assert c.top_p == 1.0


def test_env_override(monkeypatch):
    monkeypatch.setenv("RETRIEVER_AGENT_MAX_ROUNDS", "2")
    assert AgentConfig().max_rounds == 2
