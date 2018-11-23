import click

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

from . import cli

################################
#
# a lot of the provided functions are temporary
# and BEDPE format handling and sanitation should be 
# moved elsewhere 
# - bioframe ?
# -clodius style  ? - ask column number for each reuired field ...
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
               dots['chrom1'].unique()#.sort_values().values
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
    help="A threshold radius for calling a pair of"
         " identical. [Has no effect right now]"
         " euivalent to following for now:"
         " 0.2*dist if dist < 50kb else 50kb,"
         " where dist = |bin1-id-name - bin2-id-name|",
         #  consider substituting this with a proper threshold function
         #  0.2*dist if dist < 50000 else 50000, where dist = |bin1-id-name - bin2-id-name|
         #  use eval or numexpr or something along this lines ...
    type=int,
    default=20000,
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

    # accumulate matching/non-matching lists genome-wide:
    m1_gw, nm1_gw = [], []
    m2_gw, nm2_gw = [], []

    dg1 = dots_1.groupby('chrom1')
    dg2 = dots_2.groupby('chrom1')

    # perform comparison for common chromosomes:
    for chrom in common_chroms:
        # extract lists of dots per chromosome
        d1_chr = dg1.get_group(chrom)
        d2_chr = dg2.get_group(chrom)
        # create KDTrees using "start" coordinates...
        tree1 = cKDTree(d1_chr[[bin1_id_name,bin2_id_name]].values)
        tree2 = cKDTree(d2_chr[[bin1_id_name,bin2_id_name]].values)
        # intersect the trees/lists ...
        # mind non-commutative nature of it ...
        # for each peak from tree1, see if it has neighbours in tree2:
        m1_chr, nm1_chr = intersect_dots_sets(tree1,tree2)
        # for each peak from tree2, see if it has neighbours in tree1:
        m2_chr, nm2_chr = intersect_dots_sets(tree2,tree1)
        # just some sanity check 
        assert len(m1_chr)+len(nm1_chr) == len(d1_chr)
        assert len(m2_chr)+len(nm2_chr) == len(d2_chr)
        # append actual dots to the genome-wide lists:
        m1_gw.append(d1_chr.iloc[m1_chr])
        nm1_gw.append(d1_chr.iloc[nm1_chr])
        m2_gw.append(d2_chr.iloc[m2_chr])
        nm2_gw.append(d2_chr.iloc[nm2_chr])

    # concat those chrom-groups into genome-wide ones:
    m1_gw = pd.concat(m1_gw,ignore_index=True)
    nm1_gw = pd.concat(nm1_gw,ignore_index=True)
    m2_gw = pd.concat(m2_gw,ignore_index=True)
    nm2_gw = pd.concat(nm2_gw,ignore_index=True)

    # generate output describing # of matches/non-matches per chrom
    output_df = pd.DataFrame({
                    "chrom":common_chroms,
                    "1in2":m1_gw.groupby('chrom1').size().loc[common_chroms].values,
                    "2in1":m2_gw.groupby('chrom1').size().loc[common_chroms].values,
                    "1notin2":nm1_gw.groupby('chrom1').size().loc[common_chroms].values,
                    "2notin1":nm2_gw.groupby('chrom1').size().loc[common_chroms].values,
                    "1total":dg1.size().loc[common_chroms].values,
                    "2total":dg2.size().loc[common_chroms].values })

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

