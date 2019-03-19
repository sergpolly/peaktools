import pandas as pd
import numpy as np
from scipy.spatial import cKDTree


def matching_dots_tree(tin,
                        tref,
                        get_identity_thresold,
                        **kwargs):
    """A Function that returns indices of
    2D peaks from 'tin' that have a match in
    'tref', and indices of non-matching ones.

    The comparison is not commutative, thus we
    need to call one of the sets a "reference".


    Parameters
    ----------
    tin : KDTree
        KDTree of dots to be compared with
        the "reference" set tref
    tref : KDTree
        KDTree of dots that serves as "reference"
        for comparison.
    get_identity_thresold : float -> float
        A function that returns a distance threshold
        to consider a pair of dots indistinguishable
        for a given ditance from the diagonal.
    kwargs : doct
        Named parameters to provide to the
        'get_identity_thresold' function

    Returns
    -------
    matching, nonmatching : tuple of lists
        Lists with the indices of dots from tin
        that have matching dots from tref, and
        indices of tin-dots that do not.

    """

    # consider abstracting KDTree from the main script.
    # i.e. tin, tref - should become dataframes and turned
    # into trees here , maybe ...

    # go over dots from "tin" and
    # look for
    matching = []
    nonmatching = []
    idx = 0
    for peak in tin.data:
        # how far is this peak from diagonal:
        dist = np.abs(peak[0]-peak[1])
        r_threshold = get_identity_thresold(dist, **kwargs)
        # calculate # of neighbours of a peak
        # in the 'tref'-tree for a given "identity radius",
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



def intersect_dots_genomewide(dots_1,
                            dots_2,
                            common_chroms,
                            get_identity_thresold,
                            bin1_id_name='start1',
                            bin2_id_name='start2',
                            **kwargs):
    """A Function that returns indices of
    2D peaks from 'tin' that have a match in
    'tref', and indices of non-matching ones.

    The comparison is not commutative, thus we
    need to call one of the sets a "reference".


    Parameters
    ----------
    dots_1 : DataFrame
        DataFrame of dots to be compared with
        the "reference" set tref
    dots_2 : DataFrame
        DataFrame of dots that serves as "reference"
        for comparison.
    common_chroms: iterable
        An interable of common chromosomes to
        go through.
    get_identity_thresold : float -> float
        A function that returns a distance threshold
        to consider a pair of dots indistinguishable
        for a given ditance from the diagonal.
    bin1_id_name: str
        Name of the 1st coordinate (row index) to use
         for distance calculations and clustering
         alternatives include: end1, cstart1 (centroid).
    bin2_id_name: str
        Name of the 2st coordinate (row index) to use
         for distance calculations and clustering
         alternatives include: end1, cstart1 (centroid).
    kwargs : dict
        Named parameters to provide to the
        'get_identity_thresold' function

    Returns
    -------
    m1_gw, nm1_gw, m2_gw, nm2_gw : tuple of DataFrames
        tuple of genome-wide DataFrames of
        matching/non-matching dots:
        m1 - dots from 1, that have a match in 2
        nm1 - dots from 1, without a match in 2
        m2 - dots from 2, that have a match in 1
        nm2 - dots from 2, without a match in 1


    """

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
        m1_chr, nm1_chr = matching_dots_tree(tree1,tree2,dynamic_id_thresold,**kwargs)
        # for each peak from tree2, see if it has neighbours in tree1:
        m2_chr, nm2_chr = matching_dots_tree(tree2,tree1,dynamic_id_thresold,**kwargs)
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


    # return matching/nonmatching dots:
    return m1_gw, nm1_gw, m2_gw, nm2_gw


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
def dynamic_id_thresold(dist,
                        dist_ratio=0.2,
                        dist_threshold=50000,
                        padding=1000):
    """An example of dynamic identity theshold
    distance. Used in the original HiCCUPS to
    identify indistinuishable "dots".

    This threshold depends on the distnace from
    the diagonal, i.e. genomic separation that
    the compared dots are connecting.

    for small separation |x-y|>50kb:
        0.2*|x-y| + small_padding
    for separation |x-y|>50kb:
        a distance independant 50kb

    Parameters
    ----------
    dist : int or float
        A genomic separation in basepairs.
         i.e. |locus_x - locus_y| for a given dot:
         looping interactiooon between locus_x
         and locus_y.
    dist_ratio : float
        A multiplicative factor that turns genomic
        separation into an approproate threshold:
        0 < dist_ratio < 1
    dist_threshold : int or float
        Genomic separation threshold, above which
        to apply a constant, distnace independent
        threshold: dist_threshold
    padding : int or float
        Small padding for the dynamic part of
        threshold, to make sure the threshold is
        always >0.

    Returns
    -------
    float, the identity threshold

    """

    if (dist < dist_threshold):
        return dist_ratio*dist+padding 
    else:
        return dist_threshold



def const_id_thresold(dist,
                        dist_threshold=50000):
    """An example of cosnt identity theshold
    distance. Used in the original HiCCUPS to
    identify indistinuishable "dots" when
    merging lists of dots of several resolutions.

    Parameters
    ----------
    dist : int or float
        A genomic separation in basepairs.
         i.e. |locus_x - locus_y| for a given dot:
         looping interactiooon between locus_x
         and locus_y.
    dist_threshold : int or float
        Genomic separation threshold to be
        returned.

    Returns
    -------
    float, the identity threshold

    """
    return dist_threshold

