#!/usr/bin/env python

"""Tests for `synd` package."""


import unittest
import os
from click.testing import CliRunner

from synd.core import load_model
from synd import cli
import synd.hosted
from synd.models.discrete.markov import MarkovGenerator
from examples.data import simple_model
import numpy as np


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

    def test_uploading_hosted_model(self):

        access_key = os.environ.get('MINIO_ACCESSKEY')
        secret_key = os.environ.get('MINIO_SECRETKEY')

        test_object_name = 'test_synmd_model'

        client = synd.hosted.make_minio_client(access_key=access_key, secret_key=secret_key)
        synd.hosted.upload_model(self.synmd_model, test_object_name, client)

        assert client.stat_object(bucket_name=synd.hosted.MODEL_BUCKET, object_name=test_object_name) is not None

    def test_saving_loading_markov_generator(self):
        """Test saving and loading a Markov generator."""

        self.synmd_model.save("simple_synmd_model.dat")

        loaded_model = load_model("simple_synmd_model.dat")

        assert isinstance(loaded_model, MarkovGenerator)

        os.remove("simple_synmd_model.dat")

    def test_markov_trajectory_generation(self):
        """Test generating a short trajectory from a Markov generator."""

        n_steps = 10

        trajectory = self.synmd_model.generate_trajectory(
            initial_distribution=simple_model.initial_distribution,
            n_steps=n_steps
        )

        assert isinstance(trajectory, np.ndarray)
        assert trajectory.shape[0] == simple_model.initial_distribution.shape[0]
        assert trajectory.shape[1] == n_steps

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'synd.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
