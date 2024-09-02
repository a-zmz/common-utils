"""
Utility function for string manipulation.
"""
import re


def remove_punctuation(word: str):
    #return re.sub(r'[!?.:;,"()-]', "", word)
    return re.sub(r'[!?:;,"]', "", word)


def remove_bracket(word: str):
    """
    Only keep the string content before bracket.

    params
    ===
    word: str.
    """
    if '(' in word:
        return word.split('(')[0]
    else:
        return word


def swap_columns(df, swap_from, swap_to):
    """
    Swap two columns of a pandas dataframe.

    params
    ===
    df: pd dataframe.

    swap_from: str, name or partial name of the column that needs to be swapped.

    swap_to: str or int, name or index of the column that needs to be swapped to.
    """
    # get all column names
    cols = list(df.columns)

    # get index of swap from
    idx_from = [bool(re.search(swap_from, i)) for i in cols].index(1)

    # get index of swap to if it is not a str
    if isinstance(swap_to, str):
        idx_to = cols.index(swap_to)
    elif isinstance(swap_to, int):
        idx_to = swap_to
    else:
        raise TypeError(f'{swap_to} needs to be either str or int.')

    cols[idx_to], cols[idx_from] = cols[idx_from], cols[idx_to]

    # re-order columns
    df = df[cols]

    return df

# Custom sorting function
def sort_key(value):
    # Convert to string (in case it's a number)
    val_str = str(value)

    # Split the numeric part and the suffix (if any)
    if '_' in val_str:
        base, suffix = val_str.split('_')
        return (int(base), int(suffix))
    else:
        return (int(val_str), 0)
