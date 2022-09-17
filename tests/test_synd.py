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

        self.access_key = os.environ.get('MINIO_ACCESSKEY')
        self.secret_key = os.environ.get('MINIO_SECRETKEY')

        self.test_bucket = synd.hosted.MODEL_BUCKET

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_uploading_hosted_model(self):

        test_object_name = 'upload_test_model'

        client = synd.hosted.make_minio_client(access_key=self.access_key, secret_key=self.secret_key)

        synd.hosted.upload_model(model=self.synmd_model,
                                 identifier=test_object_name,
                                 client=client,
                                 bucket=self.test_bucket
                                 )

        assert client.stat_object(bucket_name=self.test_bucket, object_name=test_object_name) is not None, \
            "Model upload failed"

        client.remove_object(bucket_name=synd.hosted.MODEL_BUCKET, object_name=test_object_name)

    def test_downloading_hosted_model(self):

        test_object_name = 'download_test_model'

        client = synd.hosted.make_minio_client(access_key=self.access_key, secret_key=self.secret_key)

        assert client.stat_object(bucket_name=self.test_bucket, object_name=test_object_name) is not None, \
            "Test model doesn't exist!"

        model = synd.hosted.download_model(test_object_name, client, bucket=self.test_bucket)

        assert model is not None, "Model download failed"

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

    @unittest.skip
    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'synd.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
