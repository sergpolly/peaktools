import click



CONTEXT_SETTINGS = {
    'help_option_names': ['-h', '--help'],
}


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """
    Utilities for genomic peaks (as in bedpe files).
    """

from . import (compare_dot_lists,
    merge_dot_lists)
