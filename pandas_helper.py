import pandas as pd

from chunk_filter import ChunkFilter


def read_for_gridcell(path, lon_of_interest, lat_of_interest):
    iter_csv = pd.read_csv(path, delim_whitespace=True, iterator=True)
    return pd.concat([chunk[(chunk['Lat'] == lat_of_interest) & (chunk['Lon'] == lon_of_interest)] for chunk in iter_csv])


def read_for_years(path, year1, year2, lons_lats_of_interest=None):
    iter_csv = pd.read_csv(path, delim_whitespace=True, iterator=True)

    chunk_filter = ChunkFilter(years_of_interest=[year1, year2], lons_lats_of_interest=lons_lats_of_interest)

    if lons_lats_of_interest is not None:
        return pd.concat([chunk_filter.filter_chunk(chunk) for chunk in iter_csv])
    else:
        return pd.concat([chunk[(chunk['Year'] >= year1) & (chunk['Year'] <= year2)] for chunk in iter_csv])
