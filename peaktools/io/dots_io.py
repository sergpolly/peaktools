
import pandas as pd

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

