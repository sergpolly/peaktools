import click

# from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

from . import cli
from .io import read_validate_dots_list
from .lib import intersect_dots_genomewide, dynamic_id_thresold


@cli.command()
@click.argument(
    "dots-path-1",
    metavar="DOTS_PATH_1",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
@click.argument(
    "dots-path-2",
    metavar="DOTS_PATH_2",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
# options ...
@click.option(
    '--radius', 
    help="A threshold radius for calling a pair of"
         " identical dots."
         " equivalent to following for now:"
         " 0.2*dist+1kb if dist < radius else radius,"
         " where dist = |bin1-id-name - bin2-id-name|",
         #  consider substituting this with a proper threshold function
         #  0.2*dist if dist < 50000 else 50000, where dist = |bin1-id-name - bin2-id-name|
         #  use eval or numexpr or something along this lines ...
    type=int,
    default=50000,
    show_default=True)
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
@click.option(
    "--output", "-o",
    help="Specify output file name where to store"
         " the results of a comparison of a pair of BEDPEs."
         " Will use stdout if not provided.",
    type=str)
@click.option(
    "--out-nonoverlap1",
    help="Specify output file name where to store"
         " peaks from dots_path_1 that do not have"
         " neighbours in dots_path_2.",
    type=str)
@click.option(
    "--out-nonoverlap2",
    help="Specify output file name where to store"
         " peaks from dots_path_2 that do not have"
         " neighbours in dots_path_1.",
    type=str)
@click.option(
    "--out-overlap1",
    help="Specify output file name where to store"
         " peaks from dots_path_1 that have neighbours"
         " in dots_path_2.",
    type=str)
@click.option(
    "--out-overlap2",
    help="Specify output file name where to store"
         " peaks from dots_path_2 that have neighbours"
         " in dots_path_1.",
    type=str)
@click.option(
    "--bin1-id-name",
    help="Name of the 1st coordinate (row index) to use"
         " for distance calculations and clustering"
         " alternatives include: end1, cstart1 (centroid).",
    type=str,
    default="start1",
    show_default=True)
@click.option(
    "--bin2-id-name",
    help="Name of the 2st coordinate (column index) to use"
         " for distance calculations and clustering"
         " alternatives include: end2, cstart2 (centroid).",
    type=str,
    default="start2",
    show_default=True)


def compare_dot_lists(dots_path_1,
                      dots_path_2,
                      radius,
                      verbose,
                      output,
                      out_nonoverlap1,
                      out_nonoverlap2,
                      out_overlap1,
                      out_overlap2,
                      bin1_id_name,
                      bin2_id_name):


    # load dots lists ...
    # add some sort of cross-validation later on (kinda did) ...
    fulldots_1, dots_1, chroms_1 = read_validate_dots_list(dots_path_1,return_chroms=True)
    fulldots_2, dots_2, chroms_2 = read_validate_dots_list(dots_path_2,return_chroms=True)

    ########################
    # chrom_1 chrom_2 are sorted and uniqued ndarrays 
    # extract a list of common chroms:
    if (chroms_1 != chroms_2).any():
        print("{} and {} refers to different sets of chromosomes".format(dots_path_1,dots_path_2))
        print("chroms_1:\n{}\nchroms_2:\n{}\n".format(chroms_1,chroms_2))
        print("try proceeding with the set of common chromosomes ...")
        common_chroms = np.intersect1d(chroms_1,
                                    chroms_2,
                                    assume_unique=True)
        if len(common_chroms) == 0:
            raise ValueError("chroms intersection is empty ...")
    else:
        # if chroms are matching ...
        common_chroms = chroms_1


    if radius:
        # if cli arguments provided for the
        # identity thresholding we should report
        # them to the corresponding identity function:
        # expand on this later
        #########################################
        kwargs = dict(dist_threshold=radius)
        #########################################


    m1_gw, nm1_gw, m2_gw, nm2_gw = intersect_dots_genomewide(dots_1,
                                                            dots_2,
                                                            common_chroms,
                                                            dynamic_id_thresold,
                                                            **kwargs)
    # use .get(key,0) instead of .loc[common_chroms] to handle empty df-s:
    get_size_per_chrom = lambda df: [df.groupby("chrom1").size().get(chrom,0) \
                                                        for chrom in common_chroms]

    # generate output describing # of matches/non-matches per chrom
    output_df = pd.DataFrame({
                    "chrom":common_chroms,
                    "1in2":get_size_per_chrom(m1_gw),
                    "2in1":get_size_per_chrom(m2_gw),
                    "1notin2":get_size_per_chrom(nm1_gw),
                    "2notin1":get_size_per_chrom(nm2_gw),
                    "1total":dots_1.groupby('chrom1').size().loc[common_chroms].values,
                    "2total":dots_2.groupby('chrom1').size().loc[common_chroms].values })

    # verbose output to stdout, just genome-wide info ...
    if verbose:
        print("\nIntersecting lists 1 and 2 of dots is non-commutative\n"
            "    thus both ways will be reported.\n"
            "    Dots from 1 are reffered to as 1-s"
            " and from 2 as 2-s.\n\ngenome-wide:")
        print( output_df[["1in2",
                          "2in1",
                          "1notin2",
                          "2notin1",
                          "1total",
                          "2total"]]\
                  .sum().to_frame().T\
                  .to_csv(sep='\t',index=False))

    # output is that dataframe with match/nomatch info per chrom:
    if output:
        output_df.to_csv(output,sep='\t',index=False)
    else:
        print(output_df.to_csv(sep='\t',index=False))

    # output actual dots split into match/nomatch categories ...
    if out_nonoverlap1:
        nm1_gw.to_csv(out_nonoverlap1,sep='\t',index=False)
    if out_nonoverlap2:
        nm2_gw.to_csv(out_nonoverlap2,sep='\t',index=False)
    if out_overlap1:
        m1_gw.to_csv(out_overlap1,sep='\t',index=False)
    if out_overlap2:
        m2_gw.to_csv(out_overlap2,sep='\t',index=False)

    # return just in case ...
    return 0

