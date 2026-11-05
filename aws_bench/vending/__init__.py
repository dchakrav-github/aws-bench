"""Floci vending support for aws-bench.

Holds the Docker Compose overlay template that redirects a trial's agent and
verifier to a local Floci AWS emulator instead of real AWS. The overlay is
layered onto every trial via the runner's ``--extra-docker-compose`` option, so
the Floci wiring lives with the runner rather than being copied into each task
in the dataset.

The template is a packaged resource; load it with::

    from importlib.resources import files
    tmpl = files("aws_bench.vending").joinpath("floci-overlay.tmpl.yaml").read_text()

Render the ``__FLOCI_DNS_IP__`` / ``__FLOCI_CA_PATH__`` / ``__VEND_NET__``
placeholders (setup_vending.sh does this) before passing the rendered file to
``aws-bench run --extra-docker-compose``.
"""
