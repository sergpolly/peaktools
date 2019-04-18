
import click
import pandas as pd
import numpy as np
# from scipy.spatial import cKDTree

from . import cli

from .io import read_validate_dots_list
from .lib import intersect_dots_genomewide, const_id_thresold


@cli.command()
@click.argument(
    "dots_path_5kb",
    metavar="DOTS_PATH_5kb",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
@click.argument(
    "dots_path_10kb",
    metavar="DOTS_PATH_10kb",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
# options ...
@click.option(
    '--radius', 
    help='Radius for clustering, i.e., to consider'
         'a couple of dots "identical", typically ~20kb.',
    type=int,
    default=20000,
    show_default=True)
@click.option(
    "--strength-threshold",
    help="Threshold value for the strength filtering"
         " of unique 5kb peaks. Typically raw counts >100",
    type=int,
    default=100,
    show_default=True)
@click.option(
    "--small-threshold",
    help="Threshold value for the size filtering"
         " of unique 5kb peaks. Typically < 110kb",
    type=int,
    default=110000,
    show_default=True)
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
@click.option(
    "--output",
    help="Specify output file name where to store"
         " the results of dot-merger, in a BEDPE-like format.",
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
@click.option(
    "--strength-name",
    help="Name of the column to use as a peak strength"
         " typically it is 'count' i.e. raw pairs count.",
    type=str,
    default="count",
    show_default=True)



def merge_dot_lists_KDTree(dots_path_5kb,
                        dots_path_10kb,
                        radius,
                        strength_threshold,
                        small_threshold,
                        verbose,
                        output,
                        bin1_id_name,
                        bin2_id_name,
                        strength_name):

    # load dots lists ...
    # add some sort of cross-validation later on ...
    fulldots_5kb, dots_5kb, chroms_5kb = read_validate_dots_list(dots_path_5kb,return_chroms=True)
    fulldots_10kb, dots_10kb, chroms_10kb = read_validate_dots_list(dots_path_10kb,return_chroms=True)

    ########################
    # chrom_5kb chrom_1kb are sorted and uniqued ndarrays 
    not_matching_chroms = False
    if (chroms_5kb.shape != chroms_10kb.shape):
        not_matching_chroms = True
    elif (chroms_5kb.sort() != chroms_10kb.sort()).any():
        not_matching_chroms = True
    else:
        pass
    # extract a list of common chroms:    
    if not_matching_chroms:
        print("{} and {} refers to different sets of chromosomes".format(dots_path_1,dots_path_2))
        print("chroms_5kb:\n{}\nchroms_2:\n{}\n".format(chroms_5kb,chroms_10kb))
        print("try proceeding with the set of common chromosomes ...")
        common_chroms = np.intersect1d(chroms_5kb,
                                    chroms_10kb,
                                    assume_unique=True)
    else:
        common_chroms = chroms_5kb
    # check if common chroms list is not empty:
    if len(common_chroms) == 0:
        raise ValueError("chroms intersection is empty ...")
    # chromosomes are cleared!

    if radius:
        # if cli arguments provided for the
        # identity thresholding we should report
        # them to the corresponding identity function:
        # expand on this later
        #########################################
        kwargs = dict(dist_threshold=radius)
        #########################################


    if verbose:
        # before merging:
        print("Before merging:")
        print("number of dots_5kb: {}".format(len(dots_5kb)))
        print("number of dots_10kb: {}".format(len(dots_10kb)))
        print("")


    m5kb_gw, nm5kb_gw, m10kb_gw, nm10kb_gw = intersect_dots_genomewide(fulldots_5kb,
                                                                    fulldots_10kb,
                                                                    common_chroms,
                                                                    const_id_thresold,
                                                                    **kwargs)

    # translate to "human language":
    reproducible_5kb_dots = m5kb_gw
    reproducible_10kb_dots = m10kb_gw
    unique_5kb_dots = nm5kb_gw
    unique_10kb_dots = nm10kb_gw

    # print some intermediate message:
    if verbose:
        print(" 5kb: {} dots are reproducible, {} not, out of {}". \
                format(len(reproducible_5kb_dots),len(unique_5kb_dots),len(dots_5kb)))
        print("10kb: {} dots are reproducible, {} not, out of {}". \
                format(len(reproducible_10kb_dots),len(unique_10kb_dots),len(dots_10kb)))

    # the combined list of dots - original HiCCUPS-style:
    # is a union of reproducible_5kb_dots, unique_10kb_dots,
    # and "small" and/or "strong" ones from unique_5kb_dots ...
    unique_5kb_small = np.abs(unique_5kb_dots[bin1_id_name] - unique_5kb_dots[bin2_id_name]) <= small_threshold
    unique_5kb_strong = unique_5kb_dots[strength_name] >= strength_threshold
    unique_5kb_strong_or_small = np.logical_or(unique_5kb_small,unique_5kb_strong)

    if verbose:
        print("unique 5kb dots: {} are small, {} are strong, {} are strong||small". \
                format(unique_5kb_small.sum(),unique_5kb_strong.sum(),unique_5kb_strong_or_small.sum()))

    # now concatenate these lists ...
    dfs_to_concat = [reproducible_5kb_dots,
                     unique_10kb_dots,
                     unique_5kb_dots[unique_5kb_strong_or_small]]

    dots_merged_filtered = pd.concat(dfs_to_concat).sort_values(by=["chrom1",bin1_id_name,"chrom2",bin2_id_name])
    
    # dedup is required as overlap is unavoidable ...????
    print("DEBUG: check for duplicates:")
    print( dots_merged_filtered.duplicated(subset=["chrom1",bin1_id_name,"chrom2",bin2_id_name]) )

    if verbose:
        # final number:
        print("number of pixels after the merge: {}".format(len(dots_merged_filtered)))
        print()


    ##############################
    # OUTPUT:
    ##############################
    if output is not None:
        dots_merged_filtered.to_csv(
                                output,
                                sep='\t',
                                header=True,
                                index=False,
                                compression=None)

    #  return just in case ...
    return dots_merged_filtered






# if __name__ == '__main__':
#     merge_dot_lists()



# ###########################################
# # click example ...
# ###########################################
# # import click

# # @click.command()
# # @click.option('--count', default=1, help='Number of greetings.')
# # @click.option('--name', prompt='Your name',
# #               help='The person to greet.')
# # def hello(count, name):
# #     """Simple program that greets NAME for a total of COUNT times."""
# #     for x in range(count):
# #         click.echo('Hello %s!' % name)

# # if __name__ == '__main__':
# #     hello()




















