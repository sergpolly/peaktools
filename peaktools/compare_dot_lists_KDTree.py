import click

from scipy.spatial import KDTree, cKDTree
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from cooltools.dotfinder import clust_2D_pixels

from . import cli

################################
#
# a lot of the provided functions are temporary
# and BEDPE format handling and sanitation should be 
# moved elsewhere 
# - bioframe ?
# -clodius style  ?
#
#################################


# minimal subset of columns to handle:
must_columns = ["chrom1",
                "start1",
                "end1",
                "chrom2",
                "start2",
                "end2"]

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


def check_chr_prefix(df, columns=['chrom1','chrom2']):
    """
    Function that checks if chromosome
    labels present in a DataFrame contain
    a commonly used 'chr' prefix.
    Returns a description of the label
    situation: 'all', 'none' or 'some'.

    Parameters                                                                   
    ----------                                                                   
    df : pd.DataFrame
        DataFrame with columns containing
        chromosome labels.
    columns : iterable
        A list of column labels that contain
        chromosome labels and needs to be fixed.
        ['chrom1',chrom2] by default.

    Returns
    -------
    str, description of the label situation.

    """
    num_labels, num_prefixed_labels = 0,0
    for col in columns:
        # accumulate number of prefixed labels per column:
        num_prefixed_labels += df[col].str.startswith("chr").sum()
        # accumulate total number of labels:
        num_labels += len(df)
    # return the label situation description:
    if num_prefixed_labels == 0:
        return 'none'
    elif num_prefixed_labels < num_labels:
        return 'some'
    elif num_prefixed_labels == num_labels:
        return 'all'
    else:
        raise("Should have never happened: 'check_chr_prefix' !")



def read_validate_dots_list(dots_path,return_chroms=False):
    """
    this temporary function tries to read BEDPE,
    looks for "must_columns": chr1/2, start1/2, stop1/2
    also tries to fix the HiCCUPS BEDPE headers
    tries to fix "chrX" vs "X" chromosome label issue
    stuff like that  - go to bioframe bedpe something?

    returns full dataframe, dataframe[must_columns], (chroms)
    """
    # load dots lists ...
    dots = pd.read_table(dots_path)
    # check if 'must_columns' are present:
    if pd.Series(must_columns).isin(dots.columns).all():
        pass
    else:
        print("{} isn't in ct format, trying conversion...".format(dots_path))
        try:
            dots = dots.rename(columns=hiccups_to_cooltools)
        except KeyError as e:
            print("conversion didn't work for {} !".format(dots_path))
            raise e
    # just in case:
    dots["chrom1"] = dots["chrom1"].astype(str)
    dots["chrom2"] = dots["chrom2"].astype(str)
    # consider checking if chrom1/2 columns refer
    # to the same chroms.
    # now check chromosome labels and
    # try to fix them as needed:
    prefix_status = check_chr_prefix(dots, columns=['chrom1','chrom2'])
    if prefix_status == 'all':
        pass
    elif prefix_status == 'none':
        # add 'chr' prefix
        dots["chrom1"] = 'chr'+dots["chrom1"]
        dots["chrom2"] = 'chr'+dots["chrom2"]
    elif prefix_status == 'some':
        raise("Provided chrom labels are messed up!")
    # returning both full DataFrame and a subset
    # with must_columns only
    if return_chroms:
        return dots, \
               dots[must_columns].copy(), \
               dots['chrom1'].unique().sort_values().values
    else:
        return dots, dots[must_columns].copy()



def intersect_dots_sets(tin,
                        tref,
                        dist_threshold=50000,
                        dist_ratio=0.2,
                        padding=1000):
    """A Function that returns indices of
    2D peaks from 'tin' that have a match in
    'tref', and indices of non-matching ones.

    The comparison is not commutative, thus we
    need to call one of the sets a "reference".


    Parameters
    ----------
    tin : DataFrame
        df of dots to be compared with
        the "reference" set tref
    tref : DataFrame
        df of dots that serves as "reference"
        for comparison.
    dist_threshold : int
        Distance threshold for the
        proximity thresholding HiCCUPS-style
        to be replaced with supplying a function
    dist_threshold : float
        Distance ratio for the
        proximity thresholding HiCCUPS-style
        to be replaced with supplying a function
    padding : int
        padding for the
        proximity thresholding HiCCUPS-style
        to be replaced with supplying a function

    """

    # that's the function for proximity thresholding:
    # it yields a distance threshold used later on to
    # call a pair of dots from different lists "identical".
    #
    # This threshold depends on how far the "tin" dot
    # is from the diagonal,
    # dist = |tin_dot.x - tin_dot.y|
    #
    # we should be able to simply pass the whole
    # function as an argument, I guess ...
    #
    # what this does is simply sets a dynamic threshold
    # on what pair of dots to consider "identical" from
    # 2 lists: when tin-dot is more than "dist_threshold"
    # away from diagonal, use "dist_threshold" as a proximity
    # threhsold, othrewise use "dist_ratio"*tin_dot.x - tin_dot.y|
    # in practice - for dots close to diagonal use
    # small threshold, once more than 50kb away from diag, use
    # 50kb as a threshold ...
    get_thresold = lambda dist: dist_ratio*dist+padding \
                        if (dist < dist_threshold) \
                            else dist_threshold+padding
    # go over dots from "tin" and
    # look for
    matching = []
    nonmatching = []
    idx = 0
    for peak in tin.data:
        # how far is this peak from diagonal:
        dist = np.abs(peak[0]-peak[1])
        r_threshold = get_thresold(dist)
        # calculate # of neighbours of a peak
        # in the 'tref'-tree:
        # here is a KDTree trick:
        nneighbours= len(tref.query_ball_point(peak,r_threshold))
        # store the results either in matching or
        # nonmatching lists:
        if nneighbours > 0:
            matching.append(idx)
        elif nneighbours == 0:
            nonmatching.append(idx)
        else:
            raise("Major bug in 'intersect_dots_sets'!")
        # index increment:
        idx += 1
    return matching, nonmatching




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
    fulldots_1, dots_1, chroms_1 = read_validate_dots_list_cmp(dots_path_1,return_chroms=True)
    fulldots_2, dots_2, chroms_2 = read_validate_dots_list_cmp(dots_path_2,return_chroms=True)

    ########################
    # chrom_1 chrom_2 are sorted and uniqued ndarrays 
    #
    #
    #
    #
    #########################
    # extract a list of common chroms:
     if chroms_1 != chroms_2:
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

######################
# fixed up to here
#
# to be continiued ...

#############################################################
#############################################################
#############################################################
#############################################################
t2_ref_msg = ""
t1_ref_msg = ""

all_match = 0

m1all = pd.Index([],dtype=int)
nm1all = pd.Index([],dtype=int)
m2all = pd.Index([],dtype=int)
nm2all = pd.Index([],dtype=int)

# perform comparison for common chromosomes:
for chrom in common_chroms:
    non_matching = []
    # extract lists of dots per chromosome
    d1c = dots_1[dots_1['chrom1']==chrom]
    d2c = dots_2[dots_2['chrom1']==chrom]
    # create KDTrees using "start" coordinates...
    t1 = cKDTree(d1c[['start1','start2']].values)
    t2 = cKDTree(d2c[['start1','start2']].values)
    # intersect the trees/lists ...
    m1,nm1 = intersect_dots_sets(t1,t2)
    m2,nm2 = intersect_dots_sets(t2,t1)
    print("{}:\t{}\tt1-s not in t2, {}\tare; {}\tt2-s are in t1, {}\tare not"\
          .format(chrom,len(nm1),len(m1),len(m2),len(nm2)))
    assert len(m1+nm1) == len(d1c)
    assert len(m2+nm2) == len(d2c)
    m1all = m1all.append(d1c.index[m1])
    nm1all = nm1all.append(d1c.index[nm1])
    m2all = m2all.append(d2c.index[m2])
    nm2all = nm2all.append(d2c.index[nm2])
print("{}:\t{}\tt1-s not in t2, {}\tare; {}\tt2-s are in t1, {}\tare not"\
      .format("ALL",(nm1all).size,(m1all).size,len(m2all),len(nm2all)))
    

# assign diags:    
d1f['diag'] = (d1f['start1']-d1f['start2']).abs()
d2f['diag'] = (d2f['start1']-d2f['start2']).abs()
#############################################################
#############################################################
#############################################################
#############################################################


    # looks like lists of dots are good to go:
    if verbose:
        # before merging:
        print("Before comparison:")
        print("number of dots_1: {}\n".format(len(dots_1)))
        print("number of dots_2: {}\n".format(len(dots_2)))

    # add label to each DataFrame:
    dots_1["dot_label"] = "cmp1"
    dots_2["dot_label"] = "cmp2"

    # merge 2 DataFrames and sort (why sorting ?! just in case):
    dots_merged = pd.concat([dots_1,dots_2], ignore_index=True) \
                    .sort_values(by=["chrom1",bin1_id_name,"chrom2",bin2_id_name])

    very_verbose = False
    pixel_clust_list = []
    # clustering is done on a per-chromosome basis ...
    for chrom in chroms:
        pixel_clust = clust_2D_pixels(dots_merged[(dots_merged['chrom1']==chrom) & \
                                                  (dots_merged['chrom2']==chrom)],
                                      threshold_cluster=radius,
                                      bin1_id_name=bin1_id_name,
                                      bin2_id_name=bin2_id_name,
                                      clust_label_name='c_label_merge',
                                      clust_size_name='c_size_merge',
                                      verbose=very_verbose)
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
        # report larger >2 clusters.
        # These are a bit of an artifact, where a peak from
        # 1 list have more than one good counterpart in the
        # other list.
        print("\nNumber of pixels in unwanted >2 clusters: {}\n" \
            .format(len(dots_merged[dots_merged["c_size_merge"]>2])))


    # introduce unqie genome-wide labels per merged cluster,
    # just in case:
    dots_merged["c_label_merge"] = dots_merged["chrom1"]+ \
                                      "_" + dots_merged["c_label_merge"].astype(np.str)

    # all we need to do is to count reproducible peaks ...
    reproducible_peaks = dots_merged[dots_merged["c_size_merge"]>1]
    nonreproducible_peaks = dots_merged[dots_merged["c_size_merge"]==1]
    # number to print out ...
    number_of_reproducible_peaks = len(reproducible_peaks["c_label_merge"].unique())
    #
    if verbose:
        # describe each category:
        print("number_of_reproducible_peaks: {}\n".format(number_of_reproducible_peaks))

    ###################################################################
    # extract and output reproducible/nonoverlaping peaks ...
    ###################################################################
    rep_dots_1 = reproducible_peaks[reproducible_peaks["dot_label"]=="cmp1"]
    rep_dots_2 = reproducible_peaks[reproducible_peaks["dot_label"]=="cmp2"]
    # we are very interested in the non-reproducible peaks as well,
    # at least for debugging purposes, thus we'd want to output them ...
    nonrep_dots_1 = nonreproducible_peaks[nonreproducible_peaks["dot_label"]=="cmp1"]
    nonrep_dots_2 = nonreproducible_peaks[nonreproducible_peaks["dot_label"]=="cmp2"]

    # output:
    if out_nonoverlap1:
        nonrep_dots_1.merge(fulldots_1,
            how="left",
            # consider modifying 'must_columns_cmp' or this on
            # argument to something smaller later on ...
            on=must_columns_cmp,
            sort=True ).to_csv(out_nonoverlap1,sep='\t',index=False)
    if out_nonoverlap2:
        nonrep_dots_2.merge(fulldots_2,
            how="left",
            # consider modifying 'must_columns_cmp' or this on
            # argument to something smaller later on ...
            on=must_columns_cmp,
            sort=True ).to_csv(out_nonoverlap2,sep='\t',index=False)
    if out_overlap1:
        rep_dots_1.merge(fulldots_1,
            how="left",
            # consider modifying 'must_columns_cmp' or this on
            # argument to something smaller later on ...
            on=must_columns_cmp,
            sort=True ).to_csv(out_overlap1,sep='\t',index=False)
    if out_overlap2:
        rep_dots_2.merge(fulldots_2,
            how="left",
            # consider modifying 'must_columns_cmp' or this on
            # argument to something smaller later on ...
            on=must_columns_cmp,
            sort=True ).to_csv(out_overlap2,sep='\t',index=False)

    # return just in case ...
    return number_of_reproducible_peaks



# if __name__ == '__main__':
#     compare_dot_lists()

















