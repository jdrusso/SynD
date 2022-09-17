#!/usr/bin/env python

"""Tests for `synd` package."""


import unittest
import os
from click.testing import CliRunner

from synd.core import load_model
from synd import cli
from synd.models.discrete.markov import MarkovGenerator
from examples.data import simple_model


class TestSynd(unittest.TestCase):
    """Tests for `synd` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

        self.synmd_model = MarkovGenerator(
            transition_matrix=simple_model.transition_matrix,
            backmapper=simple_model.backmapper
        )

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_saving_markov_generator(self):
        """Test something."""

        self.synmd_model.save("simple_synmd_model.dat")

        loaded_model = load_model("simple_synmd_model.dat")

        assert isinstance(loaded_model, MarkovGenerator)

        os.remove("simple_synmd_model.dat")

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'synd.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
