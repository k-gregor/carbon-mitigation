class ChunkFilter:
    def __init__(self, years_of_interest=None, lat_of_interest=None, lon_of_interest=None, lons_lats_of_interest=None):

        assert (lon_of_interest is None and lat_of_interest is None) or (lon_of_interest is not None and lat_of_interest is not None)

        self.years_of_interest = years_of_interest
        self.lat_of_interest = lat_of_interest
        self.lon_of_interest = lon_of_interest
        self.lons_lats_of_interest = lons_lats_of_interest

    def filter_chunk(self, ccc):
        keep = ccc
        if self.years_of_interest:
            if len(self.years_of_interest) == 1:
                keep = keep[(keep['Year'] == self.years_of_interest[0])]
            if len(self.years_of_interest) == 2:
                keep = keep[(keep['Year'] >= self.years_of_interest[0]) & (keep['Year'] <= self.years_of_interest[1])]
        if self.lons_lats_of_interest is not None:
            keep = keep[keep[['Lon', 'Lat']].apply(tuple, axis=1).isin(self.lons_lats_of_interest)]
        if self.lat_of_interest is not None:
            keep = keep[(keep['Lat'] == self.lat_of_interest)]
        if self.lon_of_interest is not None:
            keep = keep[(keep['Lon'] == self.lon_of_interest)]
        return keep