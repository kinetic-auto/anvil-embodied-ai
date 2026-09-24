"""Channel matching for conversion debug plots."""

from mcap_converter.utils.debug_plot import overlay_channels


def test_effort_obs_vs_pos_effort_action_matches_by_name():
    obs = [f"right_joint{i}.effort" for i in range(1, 8)]
    act = [f"right_joint{i}.position" for i in range(1, 8)] + obs
    channels = overlay_channels(obs, act, n_obs=7, n_act=14)

    assert [c[0] for c in channels] == act
    pos_rows = channels[:7]
    effort_rows = channels[7:]
    assert all(obs_i is None and act_i == i for i, (_, obs_i, act_i) in enumerate(pos_rows))
    assert all(
        obs_i == i and act_i == i + 7
        for i, (_, obs_i, act_i) in enumerate(effort_rows)
    )


def test_identical_packed_names_overlay_all():
    names = [
        "right_joint1.position",
        "right_joint1.effort",
    ]
    channels = overlay_channels(names, names, n_obs=2, n_act=2)
    assert channels == [
        ("right_joint1.position", 0, 0),
        ("right_joint1.effort", 1, 1),
    ]


def test_images_only_plots_action_channels():
    act = [f"right_joint{i}.position" for i in range(1, 3)]
    channels = overlay_channels([], act, n_obs=0, n_act=2)
    assert channels == [
        ("right_joint1.position", None, 0),
        ("right_joint2.position", None, 1),
    ]


def test_equal_length_without_names_pairs_by_index():
    channels = overlay_channels([], [], n_obs=7, n_act=7)
    assert channels == [(f"joint_{i}", i, i) for i in range(7)]
