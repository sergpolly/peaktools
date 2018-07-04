
import click
import pandas as pd
import numpy as np
from cooltools.dotfinder import clust_2D_pixels


# minimal subset of columns to handle:
must_columns = ["chrom1",
                "start1",
                "end1",
                "chrom2",
                "start2",
                "end2"]
                # "obs.raw",
                # "cstart1",
                # "cstart2",
                # "c_size",
                # "la_exp.donut.value",
                # "la_exp.vertical.value",
                # "la_exp.horizontal.value",
                # "la_exp.lowleft.value"]
                # "la_exp.donut.qval",
                # "la_exp.vertical.qval",
                # "la_exp.horizontal.qval",
                # "la_exp.lowleft.qval"]

# HiCCUPs to cooltools BEDPE renamer, just in case:
hiccups_to_cooltools = {'chr1': "chrom1",
                    'x1': "start1",
                    'x2': "end1",
                    'chr2': "chrom2",
                    'y1': "start2",
                    'y2': "end2",
                    'color': "color",
                    'o': "obs.raw",
                    'e_bl': "la_exp.lowleft.value",
                    'e_donut': "la_exp.donut.value",
                    'e_h': "la_exp.horizontal.value",
                    'e_v': "la_exp.vertical.value",
                    'fdr_bl': "la_exp.lowleft.qval",
                    'fdr_donut': "la_exp.donut.qval",
                    'fdr_h': "la_exp.horizontal.qval",
                    'fdr_v': "la_exp.vertical.qval",
                    'num_collapsed': "c_size",
                    'centroid1': "cstart1",
                    'centroid2': "cstart2",
                    'radius': "radius"}



def read_validate_dots_list(dots_path):
    # load dots lists ...
    try:
        dots = pd.read_table(dots_path)
        dots_must = dots[must_columns]
    except KeyError as exc_one:
        print("Seems like {} is not in cooltools format, trying conversion ...".format(dots_path))
        try:
            dots = dots.rename(columns=hiccups_to_cooltools)
            dots_must = dots[must_columns].copy()
            dots_must['chrom1'] = "chr"+dots_must['chrom1']
            dots_must['chrom2'] = "chr"+dots_must['chrom2']
        except KeyError as exc_two:
            print("Seems like conversion didn't work for {}".format(dots_path))
            raise exc_two

    # returning both full DataFrame and a subset
    # with must_columns only ...
    return dots, dots_must







@click.command()
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
    help='Radius for clustering, i.e., to consider'
         'a couple of dots "identical", typically ~20kb.',
    type=int,
    default=20000,
    show_default=True)
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
# @click.option(
#     "--output",
#     help="Specify output file name where to store"
#          " the results of dot-merger, in a BEDPE-like format.",
#     type=str)
@click.option(
    "--out-nonoverlap1",
    help="Specify output file name where to store"
         " peaks from dots_path_1 that do not have"
         " a good counterpart in dots_path_2.",
    type=str)
@click.option(
    "--out-nonoverlap2",
    help="Specify output file name where to store"
         " peaks from dots_path_2 that do not have"
         " a good counterpart in dots_path_1.",
    type=str)
@click.option(
    "--out-overlap1",
    help="Specify output file name where to store"
         " peaks from dots_path_1 that have a good"
         " counterpart in dots_path_2.",
    type=str)
@click.option(
    "--out-overlap2",
    help="Specify output file name where to store"
         " peaks from dots_path_2 that have a good"
         " counterpart in dots_path_1.",
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
                      # output,
                      out_nonoverlap1,
                      out_nonoverlap2,
                      out_overlap1,
                      out_overlap2,
                      bin1_id_name,
                      bin2_id_name):


    # load dots lists ...
    # add some sort of cross-validation later on (kinda did) ...
    fulldots_1, dots_1 = read_validate_dots_list(dots_path_1)
    fulldots_2, dots_2 = read_validate_dots_list(dots_path_2)

    # extract a list of chroms:
    chroms_1 = sorted(list(dots_1['chrom1'].drop_duplicates()))
    chroms_2 = sorted(list(dots_2['chrom1'].drop_duplicates()))
    if chroms_1 != chroms_2:
        print("{} and {} refers to different sets of chromosomes".format(dots_path_1,dots_path_2))
        print("chroms_1:\n{}\nchroms_2:\n{}\n".format(chroms_1,chroms_2))
        raise ValueError("chromosomes must match ...")
    else:
        # if chroms are matching ...
        chroms = chroms_1

    # looks like lists of dots are good to go:
    if verbose:
        # before merging:
        print("Before comparison:")
        print("number of dots_1: {}".format(len(dots_1)))
        print("number of dots_2: {}".format(len(dots_2)))
        print("")

    # add label to each DataFrame:
    dots_1["dot_label"] = "cmp1"
    dots_2["dot_label"] = "cmp2"

    # merge 2 DataFrames and sort (why sorting ?! just in case):
    dots_merged = pd.concat([dots_1,dots_2],
                            ignore_index=True).sort_values(by=["chrom1",bin1_id_name,"chrom2",bin2_id_name])

    very_verbose = False
    pixel_clust_list = []
    # clustering is done on a per-chromosome basis ...
    for chrom in chroms:
        pixel_clust = clust_2D_pixels(dots_merged[(dots_merged['chrom1']==chrom) & \
                                                  (dots_merged['chrom2']==chrom)],
                                      threshold_cluster= radius,
                                      bin1_id_name= bin1_id_name,
                                      bin2_id_name= bin2_id_name,
                                      clust_label_name='c_label_merge',
                                      clust_size_name='c_size_merge',
                                      verbose= very_verbose)
        pixel_clust_list.append(pixel_clust)


    # concatenate clustering results ...
    # indexing information persists here ...
    pixel_clust_df = pd.concat(pixel_clust_list, ignore_index=False)
    # pixel_clust_list
    # must be a DataFrame with the following columns:
    # ['c'+bin1_id_name, 'c'+bin2_id_name, 'c_label_merge', 'c_size_merge']
    # thus there should be no column naming conflicts downsrteam ...

    # now merge pixel_clust_df and dots_merged DataFrame (index-wise):
    # ignore suffixes=('_x','_y'), taken care of upstream.
    dots_merged =  dots_merged.merge(pixel_clust_df,
                                     how='left',
                                     left_index=True,
                                     right_index=True)

    if verbose:
        # report larger >2 clusters:
        print("\nNumber of pixels in unwanted >2 clusters: {}\n" \
            .format(len(dots_merged[dots_merged["c_size_merge"]>2])))


    # introduce unqie genome-wide labels per merged cluster,
    # just in case:
    dots_merged["c_label_merge"] = dots_merged["chrom1"]+ \
                                        "_" + \
                                        dots_merged["c_label_merge"].astype(np.str)

    # all we need to do is to count reproducible peaks ...
    reproducible_peaks = dots_merged[dots_merged["c_size_merge"]>1]
    nonreproducible_peaks = dots_merged[dots_merged["c_size_merge"]==1]
    # number to print out ...
    number_of_reproducible_peaks = len(reproducible_peaks["c_label_merge"].unique())

    rep_dots_1 = reproducible_peaks[reproducible_peaks["dot_label"]=="cmp1"]
    rep_dots_2 = reproducible_peaks[reproducible_peaks["dot_label"]=="cmp2"]

    # we are very interested in the non-reproducible peaks as well,
    # at least for debugging purposes, thus we'd want to output them ...
    nonrep_dots_1 = nonreproducible_peaks[nonreproducible_peaks["dot_label"]=="cmp1"]
    nonrep_dots_2 = nonreproducible_peaks[nonreproducible_peaks["dot_label"]=="cmp2"]

    if out_nonoverlap1:
        nonrep_dots_1.merge(fulldots_1,
            how="left",
            # consider modifying 'must_columns' or this on
            # argument to something smaller later on ...
            on=must_columns,
            sort=True ).to_csv(out_nonoverlap1,sep='\t',index=False)
    if out_nonoverlap2:
        nonrep_dots_2.merge(fulldots_2,
            how="left",
            # consider modifying 'must_columns' or this on
            # argument to something smaller later on ...
            on=must_columns,
            sort=True ).to_csv(out_nonoverlap2,sep='\t',index=False)
    if out_overlap1:
        rep_dots_1.merge(fulldots_1,
            how="left",
            # consider modifying 'must_columns' or this on
            # argument to something smaller later on ...
            on=must_columns,
            sort=True ).to_csv(out_overlap1,sep='\t',index=False)
    if out_overlap2:
        rep_dots_2.merge(fulldots_2,
            how="left",
            # consider modifying 'must_columns' or this on
            # argument to something smaller later on ...
            on=must_columns,
            sort=True ).to_csv(out_overlap2,sep='\t',index=False)

    if verbose:
        # describe each category:
        print("number_of_reproducible_peaks: {}\n".format(number_of_reproducible_peaks))

    # return just in case ...
    return number_of_reproducible_peaks



if __name__ == '__main__':
    compare_dot_lists()

















