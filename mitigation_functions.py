import pandas as pd
from pandas import testing as tm
import numpy as np


def get_new_total_mitigation(cflux, cpool, forest_harvest, base_year = 2020, end_year=2100, percentage_decarbonization_by_2050=None):

    cflux = cflux.set_index(['Lon', 'Lat', 'Year'])
    cpool = cpool.set_index(['Lon', 'Lat', 'Year'])
    forest_harvest = forest_harvest.set_index(['Lon', 'Lat', 'Year'])

    forest_harvest['Slow_h'] = cflux['Slow_h']+ cflux['LU_ch']
    forest_harvest['c_storage'] = cpool['Total']
    forest_harvest['c_storage_veg'] = cpool['VegC']
    forest_harvest['c_storage_litter'] = cpool['LitterC']
    forest_harvest['c_storage_soil'] = cpool['SoilC']
    forest_harvest['c_storage_products'] = cpool['HarvSlowC']
    forest_harvest['c_storage_forest'] = cpool['Total'] - cpool['HarvSlowC']

    discounting_factors = None

    if percentage_decarbonization_by_2050 is not None:
        discounting_factors = get_discounting_factors_exp(end_year, percentage_decarbonization_by_2050, base_year)
        discounting_factors = discounting_factors.loc[base_year:].copy()

    bern1 = forest_harvest.reset_index()
    bernd = bern1.groupby(['Lon', 'Lat'], as_index=True).apply(
        lambda group: compute_mitigation_earths_future(group, discounting_factors, base_year))
    return bernd.reset_index()


def get_discounting_factors_exp(end_year, percentage_decarbonization_by_2050, sim_output_start):

    exp_decays = {
        1.0: -0.1535056728662697,
        0.75: -0.046209812037329684,
        0.5: -0.023104906018664842,
        0.25: -0.009589402415059362,
        0.0: 0.0,
    }

    assert percentage_decarbonization_by_2050 in exp_decays.keys(), "picked a non-supported decarbonization pace: " + str(percentage_decarbonization_by_2050)

    discounting_factors = pd.DataFrame(columns=['Year', 'factor'])
    discounting_factors['Year'] = np.linspace(sim_output_start, 2100, num=(2100 - sim_output_start + 1), dtype=int)
    discounting_factors['x'] = discounting_factors['Year'] - sim_output_start
    discounting_factors['factor'] = 1.0
    discounting_factors.loc[(discounting_factors['Year'] >= sim_output_start), 'factor'] = np.exp(exp_decays[percentage_decarbonization_by_2050]*np.arange(0, (2100 - sim_output_start + 1)))
    discounting_factors.loc[(discounting_factors['factor'] < 0), 'factor'] = 0.0
    discounting_factors.loc[(discounting_factors['factor'] > 1), 'factor'] = 1.0
    discounting_factors = discounting_factors[discounting_factors['Year'] <= end_year]
    return discounting_factors.set_index('Year')


def compute_mitigation_earths_future(forest_harvest, discounting_factors, base_year = 2020):

    forest_harvest = forest_harvest.set_index('Year')

    harvested_fuel_wood = forest_harvest['totharv_toflux']
    harvested_products = forest_harvest['totharv_toprod']
    # need to get the short pool out of the total product pool as the substitution factor is not available for the short pool in Knauf2015
    harvested_shortlived_products = forest_harvest['totharv_tosh']

    # the Knauf value does not contain end of life burning of material in the substitution factor!
    # 1.5: wood usage in Germany Knauf2015, cf. factors on average 2.1 for Sathre2010
    # Landfilling is not a thing in Germany, especially not for wood (https://www.eea.europa.eu/ims/diversion-of-waste-from-landfill)
    material_substitution_factor = 1.5
    energy_substitution_factor = 0.67  # Knauf2015

    forest_harvest['material_substitution'] = (harvested_products - harvested_shortlived_products) * material_substitution_factor

    # assume energy recovery for products after end of life
    decayed_product_pools = forest_harvest['Slow_h']
    forest_harvest['fuel_substitution'] = (harvested_fuel_wood + decayed_product_pools) * energy_substitution_factor

    if discounting_factors is not None:
        tm.assert_index_equal(forest_harvest.index, discounting_factors.index)
        forest_harvest['fuel_substitution'] *= discounting_factors['factor']
        forest_harvest['material_substitution'] *= discounting_factors['factor']

    forest_harvest['acc_fuel_substitution'] = forest_harvest['fuel_substitution'].cumsum()
    forest_harvest['fuel_mitigation'] = forest_harvest['acc_fuel_substitution'] - forest_harvest.loc[base_year, 'acc_fuel_substitution']

    forest_harvest['acc_material_substitution'] = forest_harvest['material_substitution'].cumsum()
    forest_harvest['material_mitigation'] = forest_harvest['acc_material_substitution'] - forest_harvest.loc[base_year, 'acc_material_substitution']

    forest_harvest['acc_cumulative_harvests'] = forest_harvest['totharv_toprod'].cumsum()
    forest_harvest['cumulative_harvests'] = forest_harvest['acc_cumulative_harvests'] - forest_harvest.loc[base_year, 'acc_cumulative_harvests']

    forest_harvest['cstorage_mitigation'] = forest_harvest['c_storage'] - forest_harvest.loc[base_year, 'c_storage']
    forest_harvest['cstorage_mitigation_products'] = forest_harvest['c_storage_products'] - forest_harvest.loc[base_year, 'c_storage_products']
    forest_harvest['cstorage_mitigation_forest'] = forest_harvest['c_storage_forest'] - forest_harvest.loc[base_year, 'c_storage_forest']

    forest_harvest['total_mitigation'] = forest_harvest['cstorage_mitigation'] + forest_harvest['material_mitigation'] + forest_harvest['fuel_mitigation']

    # for some reason Lon and Lat are in the index _and_ in the columns after this, so we delete the columns here...
    del forest_harvest['Lon']
    del forest_harvest['Lat']

    return forest_harvest