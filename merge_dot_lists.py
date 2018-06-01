
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
                "end2",
                "obs.raw",
                "cstart1",
                "cstart2",
                "c_label",
                "c_size",
                "la_exp.donut.value",
                "la_exp.vertical.value",
                "la_exp.horizontal.value",
                "la_exp.lowleft.value"]
                # "la_exp.donut.qval",
                # "la_exp.vertical.qval",
                # "la_exp.horizontal.qval",
                # "la_exp.lowleft.qval"]


def read_validate_dots_list(dots_path):
    # load dots lists ...
    dots = pd.read_table(dots_path)

    try:
        dots_must = dots[must_columns]
    except KeyError as exc_one:
        print("Seems like {} is not in cooltools format or lacks some columns ...".format(dots_path))
        raise exc_one

    # returning the subset:
    return dots_must





@click.command()
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
    "--bin1_id_name",
    help="Name of the 1st coordinate (row index) to use"
         " for distance calculations and clustering"
         " alternatives include: end1, cstart1 (centroid).",
    type=str,
    default="start1",
    show_default=True)
@click.option(
    "--bin2_id_name",
    help="Name of the 2st coordinate (column index) to use"
         " for distance calculations and clustering"
         " alternatives include: end2, cstart2 (centroid).",
    type=str,
    default="start2",
    show_default=True)



def merge_dot_lists(dots_path_5kb,
                    dots_path_10kb,
                    radius,
                    verbose,
                    output,
                    bin1_id_name,
                    bin2_id_name):


    # load dots lists ...
    dots_5kb  = read_validate_dots_list(dots_path_5kb)
    dots_10kb = read_validate_dots_list(dots_path_10kb)

    if verbose:
        # before merging:
        print("Before merging:")
        print("number of dots_5kb: {}".format(len(dots_5kb)))
        print("number of dots_10kb: {}".format(len(dots_10kb)))
        print("")


    # add some sort of cross-validation later on:

    # add label to each DataFrame:
    dots_5kb["res"] = "5kb"
    dots_10kb["res"] = "10kb"

    # extract a list of chroms:
    chroms = list(dots_5kb['chrom1'].drop_duplicates())

    # merge 2 DataFrames and sort (why sorting ?! just in case):
    dots_merged = pd.concat([dots_5kb,dots_10kb],
                            ignore_index=True).sort_values(by=["chrom1",bin1_id_name,"chrom2",bin2_id_name])

    # l10dat[["chrom1","start1","end1","chrom2","start2","end2"]].copy()

    pixel_clust_list = []
    for chrom in chroms:
        pixel_clust = clust_2D_pixels(dots_merged[(dots_merged['chrom1']==chrom) & \
                                                  (dots_merged['chrom2']==chrom)],
                                      threshold_cluster = radius,
                                      bin1_id_name      = bin1_id_name,
                                      bin2_id_name      = bin2_id_name,
                                      verbose = verbose)
        pixel_clust_list.append(pixel_clust)

    # concatenate clustering results ...
    # indexing information persists here ...
    pixel_clust_df = pd.concat(pixel_clust_list, ignore_index=False)
    # now merge pixel_clust_df and dots_merged DataFrame ...
    # # and merge (index-wise) with the main DataFrame:
    dots_merged =  dots_merged.merge(
                                    pixel_clust_df,
                                    how='left',
                                    left_index=True,
                                    right_index=True,
                                    suffixes=('', '_merge'))

    if verbose:
        # report >2 clusters:
        print()
        print("Number of pixels in unwanted >2 clusters: {}".format(len(dots_merged[dots_merged["c_size_merge"]>2])))
        print()


    # next thing we should do is to remove
    # redundant peaks called at 10kb, that were
    # also called at 5kb (5kb being a priority) ... 

    # there will be groups (clusters) with > 2 pixels
    # i.e. several 5kb and 10kb pixels combined
    # for now let us keep 5kb, if it's alone or
    # 5kb with the highest obs.raw ...


    # introduce unqie label per merged cluster, just in case:
    dots_merged["c_label_merge"] = dots_merged["chrom1"]+"_"+dots_merged["c_label_merge"].astype(np.str)

    # now let's just follow HiCCUPs filtering process:
    all_5kb_peaks = dots_merged[dots_merged["res"] == "5kb"]
    unique_5kb_peaks = all_5kb_peaks[all_5kb_peaks["c_size_merge"] == 1]
    all_10kb_peaks = dots_merged[dots_merged["res"] == "10kb"]
    # 1. extract only reproducible 5kb peaks :
    reproducible_5kb_peaks = all_5kb_peaks[ all_5kb_peaks["c_size_merge"]>1 ]
    # 2. extract unique 10Kb peaks :
    unique_10kb_peaks = all_10kb_peaks[ all_10kb_peaks["c_size_merge"]==1 ]
    # 3. extract unique 5kb peaks close to diagonal :
    diagonal_distance_5kb_peak = np.abs(unique_5kb_peaks[bin1_id_name] - unique_5kb_peaks[bin2_id_name])
    unique_5kb_peaks_around_diagonal = unique_5kb_peaks[ diagonal_distance_5kb_peak < 110000 ]
    # 4. extract unique 5kb peaks that appear particularly strong :
    strength_5kb_peak = unique_5kb_peaks["obs.raw"]
    unique_5kb_peaks_strong = unique_5kb_peaks[ strength_5kb_peak > 100 ]


    if verbose:
        # describe each category:
        print("number of reproducible_5kb_peaks: {}".format(len(reproducible_5kb_peaks)))
        print("number of unique_10kb_peaks: {}".format(len(unique_10kb_peaks)))
        print("number of unique_5kb_peaks_around_diagonal: {}".format(len(unique_5kb_peaks_around_diagonal)))
        print("number of unique_5kb_peaks_strong: {}".format(len(unique_5kb_peaks_strong)))
        print()



    # now concatenate these lists ...
    dfs_to_concat = [reproducible_5kb_peaks,
                     unique_10kb_peaks,
                     unique_5kb_peaks_around_diagonal,
                     unique_5kb_peaks_strong]

    dots_merged_filtered = pd.concat(dfs_to_concat).sort_values(by=["chrom1",bin1_id_name,"chrom2",bin2_id_name])
    # dedup is required as overlap is unavoidable ...
    dots_merged_filtered = dots_merged_filtered.drop_duplicates(subset=["chrom1",bin1_id_name,"chrom2",bin2_id_name])

    if verbose:
        # final number:
        print("number of pixels after the merge: {}".format(len(dots_merged_filtered)))
        print()


    ##############################
    # OUTPUT:
    ##############################
    if output is not None:
        dots_merged_filtered[must_columns].to_csv(
                                        output,
                                        sep='\t',
                                        header=True,
                                        index=False,
                                        compression=None)

    #  return just in case ...
    return dots_merged_filtered






if __name__ == '__main__':
    merge_dot_lists()



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




















