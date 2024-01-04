import mitigation_functions as mitifunc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.errors import EmptyDataError
import string

import pandas_helper as ph

partition_colors = {
    'cstorage_mitigation_forest' : '#5b8c5a',
    'cstorage_mitigation_products' : '#cfd186',
    'fuel_mitigation' : '#2e5077',
    'material_mitigation' : '#4da1a9',
}

forest_age_name = {
    1910: 'mature',
    1930: 'mature',
    1960: 'medium',
    1990: 'young',
    2000: 'young'
}

forest_type_name = {
    'ne': 'needle-leaved',
    'bd': 'broad-leaved'
}

names = {
    'total_mitigation': 'Total Forest Mitigation',
    'material_mitigation': 'Material Substitution',
    'fuel_mitigation': 'Fuel Substitution',
    'cstorage_mitigation': 'Total Carbon Sink',
    'cstorage_mitigation_products': 'Product Carbon Sink',
    'cstorage_mitigation_forest': 'Forest Carbon Sink',
}


def get_subs_data(basepath, gc, end_year=2100, dir_extension='', incdist=None, roll_window=10, base_year=2020, current_only=False, rcp45=False, middle_decarb_only=True, decarb_rate=None, forest_type=None, forest_plant_time=None):

    subs_data = pd.DataFrame(columns=['RCP', 'Man', 'Usage', 'Salvage', 'Substitution', 'cstorage_mitigation'])

    end_year_with_rolling = end_year - roll_window + 1 # e.g., 5-er window: 2020 contains 2020-2024, 2096 contains 2096-2100

    rcps = ['rcp26', 'rcp85']
    if rcp45:
        rcps.append('rcp45')

    incdist_arr = [0, 1, 2]
    if incdist is not None:
        if np.isscalar(incdist):
            incdist_arr = [incdist]
        else:
            incdist_arr = incdist


    decarb_array = [0.0, 0.25, 0.5, 0.75, 1.0]
    if middle_decarb_only:
        decarb_array = [0.25, 0.5, 0.75]
    elif decarb_rate is not None:
        decarb_array = [decarb_rate]

    forest_type_array = ['ne', 'bd'] if not forest_type else [forest_type]
    age_array = [1930, 1990] if not forest_plant_time else [forest_plant_time]

    idx = 0
    for forest_type in forest_type_array:
        for rcp in rcps:
            for usage in ([100, 150] if not current_only else [100]):
                for residence_time in ([100, 150] if not current_only else [100]):
                    for man in ([0, 50, 100, 150] if not current_only else [100]):
                        for salvage in ([0, 1] if not current_only else [1]):
                            for plant_time in age_array:
                                for incdist in incdist_arr:

                                    if man == 0 and salvage == 1:
                                        continue

                                    simname = 'forest_' + forest_type + '_rcp_' + str(rcp) + '_usage_' + str(usage) + '_residence_' + str(residence_time) + '_man_' + str(man) + '_salvage_' + str(salvage) + '_disturb_' + str(plant_time) + '_increaseddist_' + str(incdist) + dir_extension

                                    path = basepath + simname + '/'

                                    # using mean is ok, it doesnt matter that the gcs are of different size. they dont have a realistic forest share, but all 100% forest.
                                    forest_harvest = ph.read_for_years(path + 'forest_harvest.out', base_year, end_year, lons_lats_of_interest=gc).groupby('Year').mean().reset_index()
                                    cflux = ph.read_for_years(path + 'cflux_forest.out', base_year, end_year, lons_lats_of_interest=gc).groupby('Year').mean().reset_index()
                                    cpool = ph.read_for_years(path + 'cpool_forest.out', base_year, end_year, lons_lats_of_interest=gc).groupby('Year').mean().reset_index()

                                    assert len(forest_harvest[forest_harvest['Year'] == end_year]) == 1, path + ' simulation seems to not have finished'
                                    assert len(cflux[cflux['Year'] == end_year]) == 1, path + ' simulation seems to not have finished'
                                    assert len(cpool[cpool['Year'] == end_year]) == 1, path + ' simulation seems to not have finished'

                                    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=roll_window)

                                    forest_harvest = forest_harvest.set_index(['Lon', 'Lat', 'Year']).rolling(window=indexer).mean().reset_index()
                                    forest_harvest = forest_harvest[(forest_harvest['Year']>=base_year) & (forest_harvest['Year']<=end_year_with_rolling)]
                                    cflux = cflux.set_index(['Lon', 'Lat', 'Year']).rolling(window=indexer).mean().reset_index()
                                    cflux = cflux[(cflux['Year']>=base_year) & (cflux['Year']<=end_year_with_rolling)]
                                    cpool = cpool.set_index(['Lon', 'Lat', 'Year']).rolling(window=indexer).mean().reset_index()
                                    cpool = cpool[(cpool['Year']>=base_year) & (cpool['Year']<=end_year_with_rolling)]

                                    for decarconization_by_2050 in decarb_array:
                                        miti = mitifunc.get_new_total_mitigation(cflux, cpool, forest_harvest, base_year=base_year, end_year=end_year_with_rolling, percentage_decarbonization_by_2050=decarconization_by_2050)
                                        miti_grouped = miti.copy()
                                        miti_grouped = miti_grouped.set_index('Year')
                                        miti_grouped['Substitution'] = miti_grouped[['material_mitigation', 'fuel_mitigation']].sum(axis=1)

                                        new_row = pd.DataFrame({'RCP': rcp, 'Man': man, 'Usage': usage, 'Salvage': salvage, 'Disturbtime': plant_time, 'Forest': forest_type, 'Residence': residence_time, 'incdist': incdist,
                                                                'Substitution': miti_grouped.loc[end_year_with_rolling, 'Substitution'],
                                                                'cstorage_mitigation': miti_grouped.loc[end_year_with_rolling, 'cstorage_mitigation'],
                                                                'cstorage_mitigation_products': miti_grouped.loc[end_year_with_rolling, 'cstorage_mitigation_products'],
                                                                'cstorage_mitigation_forest': miti_grouped.loc[end_year_with_rolling, 'cstorage_mitigation_forest'],
                                                                'cumulative_harvests': miti_grouped.loc[end_year_with_rolling, 'cumulative_harvests'],
                                                                'fuel_mitigation': miti_grouped.loc[end_year_with_rolling, 'fuel_mitigation'],
                                                                'material_mitigation': miti_grouped.loc[end_year_with_rolling, 'material_mitigation'],
                                                                'total_miti': miti_grouped.loc[end_year_with_rolling, 'total_mitigation'],
                                                                'percentage_decarbonization_by_2050': decarconization_by_2050}, index = ['rcp_%s_man_%d_usage_%d_salvage_%d'.format(rcp, man, usage, salvage)])
                                        subs_data = pd.concat([subs_data, new_row], ignore_index=True)

    return subs_data


def plot_substitution_analysis(forest_type, subs_data=None, consider_salvaging=0, ylims=(-8, 22), plot_filepath=None):

    plt.style.use('seaborn')
    sns.set(font_scale=1.25)

    pd.options.mode.chained_assignment = 'raise'

    from matplotlib.lines import Line2D

    JITTER = 0.025

    subs_data = subs_data[(subs_data['Forest'] == forest_type)].copy()

    datasets = {'RCP 2.6': subs_data[subs_data['RCP'] == 'rcp26'].copy(), 'RCP 4.5': subs_data[subs_data['RCP'] == 'rcp45'].copy(), 'RCP 8.5': subs_data[subs_data['RCP'] == 'rcp85'].copy()}

    dotsize = 125

    for distidx, disturbtime in enumerate([1930, 1990]):

        fig, axs = plt.subplots(1, len(datasets.items()), figsize=(10 * len(datasets.items()), 6))

        for idx, (name, data) in enumerate(datasets.items()):

            data=data[data['Disturbtime'] == disturbtime]

            ax=axs[idx] if len(datasets.items()) > 1 else axs

            alpha_legend_elements = []
            color_legend_elements = []

            legend_plotted=False

            for percentage_decarbonization_by_2050 in [0.75, 0.5, 0.25]:
                for [usage_strength, residence], group in data[(data['Salvage']==consider_salvaging) & (data['percentage_decarbonization_by_2050']==percentage_decarbonization_by_2050)].groupby(['Usage', 'Residence']):

                    if usage_strength == 50:
                        continue
                    if usage_strength == 150 and residence == 100:
                        continue
                    if usage_strength == 100 and residence == 150:
                        continue

                    group.loc[(group['Man'] == 0), 'percentage_decarbonization_by_2050'] -= 2*JITTER
                    group.loc[(group['Man'] == 50), 'percentage_decarbonization_by_2050'] -= JITTER
                    group.loc[(group['Man'] == 150), 'percentage_decarbonization_by_2050'] += JITTER
                    group.loc[(group['Man'] == 200), 'percentage_decarbonization_by_2050'] += 2*JITTER
                    group.loc[:, 'percentage_decarbonization_by_2050'] *= 100

                    if usage_strength>0 and not legend_plotted:
                        legend_plotted=True

                    alpha = 1 if usage_strength == 100 else 0.33

                    group = group.reset_index()

                    sns.lineplot(data=group, x='percentage_decarbonization_by_2050', y='total_miti', legend=False, ax=ax, markers=True, dashes=False, sizes=(2.5, 2.5), alpha=0.1, color='k')
                    sns.scatterplot(data=group, x='percentage_decarbonization_by_2050', y='total_miti', color='k', alpha=alpha, ax=ax, legend=(idx==1 and not legend_plotted), s=dotsize, edgecolor='k', sizes=(10,350))

                    if percentage_decarbonization_by_2050 == 0.25:
                        alpha_legend_elements.append(Line2D([0], [0], alpha=alpha, marker='o', color='None', label=usage_strength, markerfacecolor='k', markersize=12))

                        sns.lineplot(data=group, x='percentage_decarbonization_by_2050', y='cstorage_mitigation', legend=False, ax=ax, markers=True, dashes=False, sizes=(2.5, 2.5), alpha=0.1, color='k')
                        sns.scatterplot(data=group, x='percentage_decarbonization_by_2050', y='cstorage_mitigation', color='green', alpha=alpha, ax=ax, legend=(idx==1 and not legend_plotted), s=dotsize, edgecolor='k', sizes=(10,350))

                        if usage_strength == 100:
                            sns.scatterplot(data=group[group['Man']==100], x='percentage_decarbonization_by_2050', y='total_miti', color='w', marker='X', alpha=1, ax=ax, legend=True, s=dotsize*2/3, edgecolor='k', sizes=(10,350))

                        if usage_strength == 100:
                            markersize = 18
                            color_legend_elements.append(Line2D([0], [0], alpha=1, marker='o', color='None', label='Total Mitigation', markerfacecolor='k', markersize=markersize))
                            color_legend_elements.append(Line2D([0], [0], alpha=0.33, marker='o', color='None', label='Total Mitigation (increase material usage)', markerfacecolor='k', markersize=markersize))
                            color_legend_elements.append(Line2D([0], [0], alpha=1, marker='o', color='None', label='Combined Sink', markerfacecolor='green', markersize=markersize))
                            color_legend_elements.append(Line2D([0], [0], alpha=0.33, marker='o', color='None', label='Combined Sink (increase material usage)', markerfacecolor='green', markersize=markersize))
                            color_legend_elements.append(Line2D([0], [0], alpha=1, marker='X', color='None', label='Current Mitigation Level', markeredgewidth=1, markeredgecolor='k', markerfacecolor='white', markersize=markersize))

            ax.set_xlabel('% decarbonization of other industries in 2050', fontsize=16)
            ax.set_ylabel('Cumulative Mitigation (kgC/m$^2$)', fontsize=16)

            if idx == 0 and disturbtime == 1930:
                plt.legend(handles=color_legend_elements, loc='lower right', title_fontsize=23)

            inset_y = ylims[0] + 3.5
            axins = ax.inset_axes([-2*JITTER*100 + 25, inset_y, 3*JITTER*100, 1.3], transform=ax.transData)
            axins.set_xlabel('% Harvest Intensity', fontsize=12, labelpad=-5)
            axins.set(xticks=[-0.5, 0, 0.5, 1.0])
            axins.set(xticklabels=['0', '50', '100', '150'])
            axins.set(ylabel=None)
            axins.set(yticks=[])
            axins.set(yticklabels=[])
            axins.tick_params(labelsize=11, labelrotation=-45, direction='out', length=5, width=2, zorder=300, bottom=True)
            axins.patch.set_alpha(0)
            axins.grid(False)

            axins.spines['top'].set_visible(False)
            axins.spines['right'].set_visible(False)
            axins.spines['bottom'].set_visible(False)
            axins.spines['left'].set_visible(False)

            ax.set_title(forest_age_name[disturbtime].capitalize() + ' ' + forest_type.upper() + ' forest / ' + name)
            ax.set_ylim(ylims)
            ax.set(xticks=[25, 50, 75]) # % decarb
            ax.tick_params(labelsize=17)

            ax.arrow(-2*JITTER*100 + 25, inset_y, 3.8*JITTER*100, 0, fc='k', ec='k', lw = 1,
                     head_width=0.3, head_length=1,
                     length_includes_head= True, clip_on = False, label='bernd')

        for idx2, ax in enumerate(axs):
            ax.text(-0.1, 1.1, string.ascii_lowercase[2*distidx + idx2] + ')', transform=ax.transAxes, size=20)

        plt.tight_layout()


def plot_bar_plot(subs_data, forest_type, min_y=-10, max_y=9):
    plt.style.use('seaborn')

    for disttime in [1930, 1990]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))

        for idx, rcp in enumerate(['rcp26', 'rcp85']):

            data2 = subs_data[
                (subs_data['Forest'] == forest_type) &
                (subs_data['Disturbtime'] == disttime) &
                (subs_data['RCP'] == rcp) &
                (subs_data['Salvage'] == 0) &
                (subs_data['percentage_decarbonization_by_2050']==0.0) &
                (subs_data['Man'] == 100) &
                (((subs_data['Usage']==100) & (subs_data['Residence']==100)) | ((subs_data['Usage']==150) & (subs_data['Residence']==150)))
                ].copy()

            data2['xpos'] = data2['percentage_decarbonization_by_2050']/1.2 + data2['Usage']/900

            width = 0.09
            pos_bottom = (data2['cstorage_mitigation_forest'].copy() * 0).values
            neg_bottom = (data2['cstorage_mitigation_forest'].copy() * 0).values
            net = data2['cstorage_mitigation_forest'].copy() * 0

            for column in ['cstorage_mitigation_forest', 'cstorage_mitigation_products', 'fuel_mitigation', 'material_mitigation']:
                # forest or product sink may be negative!
                if np.any(data2[column] < 0):
                    if column != 'cstorage_mitigation_forest' and column != 'cstorage_mitigation_products':
                        raise EmptyDataError()

                bottom = pos_bottom.copy()

                bottom[(data2[column] < 0)] = neg_bottom[(data2[column] < 0)]

                ax[idx].bar(data2['xpos'], data2[column], width=width, bottom=bottom, color=partition_colors[column], label=names[column])
                net += data2[column]

                neg_bottom_add = data2[column].copy()
                neg_bottom_add[(neg_bottom_add > 0)] = 0
                pos_bottom_add = data2[column].copy()
                pos_bottom_add[(pos_bottom_add < 0)] = 0

                neg_bottom += neg_bottom_add.values
                pos_bottom += pos_bottom_add.values

            ax[idx].scatter(data2['xpos'], net, color='black', marker='o', s=50, label='Net Climate Change Mitigation')

            ax[idx].set_ylim([min_y, max_y])
            ax[idx].set_title(rcp)
            ax[idx].set_xticks([1.5/9, 0.25/1.2 + 1.5/9, 0.5/1.2 + 1.5/9, 0.75/1.2 + 1.5/9, 1/1.2 + 1.5/9]) # comes from the computation of the x axis above. Middle one is just a dummy, that looks better.
            ax[idx].set_xticklabels(['0% Decarbonization', '', '50% Decarbonization', '', '100% Decarbonization'])
            ax[idx].set_ylabel('MtCO$_2$')
            ax[idx].axhline(0, color='k', linestyle='--')

        plt.suptitle(forest_age_name[disttime].capitalize() + ' ' + forest_type.upper() + ' Forest')
        plt.legend(bbox_to_anchor=(1.5, 1.0))


def plot_bar_plot_forest_types(ax, subs_data, min_y=-1000, max_y=2500, rcp='rcp26', man=100, plotaddition=None):
    plt.style.use('seaborn')

    bd_product_sinks = []
    ne_product_sinks = []

    xpos = 0
    labels = []
    for forest_type in ['ne', 'bd']:
        for disttime in [1930, 1990]:

            if man == 100000 and disttime == 1990:
                continue

            for idx, decarb in enumerate([0.25, 0.5, 0.75]):

                data2 = subs_data[
                    (subs_data['Forest'] == forest_type) &
                    (subs_data['Disturbtime'] == disttime) &
                    (subs_data['RCP'] == rcp) &
                    (subs_data['Salvage'] == 0) &
                    (subs_data['percentage_decarbonization_by_2050']==decarb) &
                    (subs_data['Man'] == man) &
                    ((subs_data['Usage'] == 100) & (subs_data['Residence'] == 100))
                    ].copy()

                data2['xpos'] = xpos
                xpos+=0.5

                labels.append(forest_age_name[disttime] + ', ' + forest_type.upper() + ', ' + str(int(decarb*100)) + '%')

                width = 0.4
                pos_bottom = (data2['cstorage_mitigation_forest'].copy() * 0).values
                neg_bottom = (data2['cstorage_mitigation_forest'].copy() * 0).values
                net = data2['cstorage_mitigation_forest'].copy() * 0

                for column in ['cstorage_mitigation_forest', 'cstorage_mitigation_products', 'fuel_mitigation', 'material_mitigation']:
                    # forest or product sink may be negative!
                    if np.any(data2[column] < 0):
                        if column != 'cstorage_mitigation_forest' and column != 'cstorage_mitigation_products':
                            raise EmptyDataError()

                    bottom = pos_bottom.copy()
                    bottom[(data2[column] < 0)] = neg_bottom[(data2[column] < 0)]

                    ax.bar(data2['xpos'], data2[column], width=width, bottom=bottom, color=partition_colors[column], label=names[column] if xpos == 1 else None)
                    net += data2[column]


                    neg_bottom_add = data2[column].copy()
                    neg_bottom_add[(neg_bottom_add > 0)] = 0
                    pos_bottom_add = data2[column].copy()
                    pos_bottom_add[(pos_bottom_add < 0)] = 0

                    neg_bottom += neg_bottom_add.values
                    pos_bottom += pos_bottom_add.values

                print('decarb', str(decarb), forest_type, ': substitution/total', data2['Substitution'].values[0]/data2['total_miti'].values[0])
                if decarb == 0.75:
                    print('Forest sink', str(data2['cstorage_mitigation_forest'].values[0]))
                    print('Product sink', str(data2['cstorage_mitigation_products'].values[0]))
                    print('Substitution', str(data2['Substitution'].values[0]))
                    print('Product/forest sink', str(data2['cstorage_mitigation_products'].values[0]/(data2['cstorage_mitigation_forest'].values[0] + data2['cstorage_mitigation_products'].values[0])))
                    if forest_type == 'bd':
                        bd_product_sinks.append(data2['cstorage_mitigation_products'])
                    else:
                        ne_product_sinks.append(data2['cstorage_mitigation_products'])

                ax.scatter(data2['xpos'], net, color='black', marker='o', s=50, label='Net Mitigation' if xpos == 1 else None, zorder=10)

    print('product sink ne/bd', np.mean(ne_product_sinks)/np.mean(bd_product_sinks))

    ax.set_ylim([min_y, max_y])
    ax.set_xticks(np.linspace(0, 5.5, 12))
    ax.set_xticklabels(labels, rotation = -25, ha='left', fontsize=11)
    ax.set_ylabel('kgC/m$^2$')
    ax.axhline(0, color='k', linestyle='--')

    ax.set_title(str(plotaddition))


def plot_violins(fig, ax, path, cells, incdist=None, end_year=2100, base_year=2019, rcp_only=None, rcp45=False, middle_decarb_only=False, ymin=None, ymax=None, forest_type=None, disturb=None, letter=None):

    CONSIDERED_C_SINK = 'cstorage_mitigation'

    subs_data_for_violins = get_subs_data(path, gc=cells, base_year=base_year, end_year=end_year, dir_extension='_newharv100patch_insdistfix', roll_window=1, incdist=incdist, rcp45=rcp45, middle_decarb_only=middle_decarb_only, forest_type=forest_type, forest_plant_time=disturb).copy()

    subs_data_for_violins2 = subs_data_for_violins.copy()
    subs_data_for_violins2 = subs_data_for_violins2[subs_data_for_violins['Salvage'] == 0]
    subs_data_for_violins = subs_data_for_violins[(subs_data_for_violins['Man'] > 0) & (subs_data_for_violins['Man'] < 100000)]
    if rcp_only:
        subs_data_for_violins = subs_data_for_violins[subs_data_for_violins['RCP'] == rcp_only]

    all_vars = ['Man', 'Usage', 'Residence', 'Disturbtime', 'percentage_decarbonization_by_2050', 'Salvage', 'Forest', 'RCP']
    all_diffs = []
    variables = []
    if not rcp_only:
        if rcp45:
            variables.append(['RCP', 'rcp26', 'rcp45', 'less climate change'])
            variables.append(['RCP', 'rcp85', 'rcp45', 'more climate change'])
        else:
            variables.append(['RCP', 'rcp26', 'rcp85', 'RCP2.6 vs RCP8.5'])

    variables.append(['percentage_decarbonization_by_2050', 0.25, 0.5, 'slower decarbonization'])
    variables.append(['percentage_decarbonization_by_2050', 0.75, 0.5, 'faster decarbonization'])
    variables.append(['Man', 150, 100, 'increased harvest intensity'])
    variables.append(['Man', 50, 100, 'decreased harvest intensity'])
    variables.append(['Salvage', 1, 0, 'execute salvage logging'])

    for variable_details in variables:

        other_vars = [ele for ele in all_vars if ele != variable_details[0]]

        variable = variable_details[0]
        first = variable_details[1]
        second = variable_details[2]
        variable_as_text = variable_details[3]

        agg1 = pd.DataFrame({
            'Total Mitigation': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[grp[variable] == first]['total_miti'].values[0] - grp[grp[variable] == second]['total_miti'].values[0]),
            'Combined Sink': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[grp[variable] == first][CONSIDERED_C_SINK].values[0] - grp[grp[variable] == second][CONSIDERED_C_SINK].values[0]),
            'Carbon Sink A': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[grp[variable] == first][CONSIDERED_C_SINK].values[0]),
            'Carbon Sink B': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[grp[variable] == second][CONSIDERED_C_SINK].values[0]),
            'Miti A': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[grp[variable] == first]['total_miti'].values[0]),
            'Miti B': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[grp[variable] == second]['total_miti'].values[0])
        }).melt(value_vars=['Total Mitigation', 'Combined Sink'], id_vars=['Carbon Sink A', 'Carbon Sink B', 'Miti A', 'Miti B'], var_name='Mitigation Type', ignore_index=False)
        agg1['variable'] = variable_as_text

        all_diffs.append(agg1)

    other_vars = [ele for ele in all_vars if (ele != 'Usage' and ele != 'Residence')]
    diffs_material_residence_increase = pd.DataFrame({
        'Total Mitigation': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[(grp['Usage'] == 150) & (grp['Residence'] == 150)]['total_miti'].values[0] - grp[(grp['Usage'] == 100) & (grp['Residence'] == 100)]['total_miti'].values[0]),
        'Combined Sink': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[(grp['Usage'] == 150) & (grp['Residence'] == 150)][CONSIDERED_C_SINK].values[0] - grp[(grp['Usage'] == 100) & (grp['Residence'] == 100)][CONSIDERED_C_SINK].values[0]),
        'Carbon Sink A': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[(grp['Usage'] == 150) & (grp['Residence'] == 150)][CONSIDERED_C_SINK].values[0]),
        'Carbon Sink B': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[(grp['Usage'] == 100) & (grp['Residence'] == 100)][CONSIDERED_C_SINK].values[0]),
        'Miti A': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[(grp['Usage'] == 150) & (grp['Residence'] == 150)]['total_miti'].values[0]),
        'Miti B': subs_data_for_violins.groupby(other_vars).apply(lambda grp : grp[(grp['Usage'] == 100) & (grp['Residence'] == 100)]['total_miti'].values[0]),
    }).melt(value_vars=['Total Mitigation', 'Combined Sink'], id_vars=['Carbon Sink A', 'Carbon Sink B', 'Miti A', 'Miti B'], var_name='Mitigation Type', ignore_index=False)
    diffs_material_residence_increase['variable'] = 'increased material usage'

    other_vars = [ele for ele in all_vars if ele != 'Man']
    diffs_unmanaged = pd.DataFrame({
        'Total Mitigation': subs_data_for_violins2.groupby(other_vars).apply(lambda grp : grp[grp['Man'] == 0]['total_miti'].values[0] - grp[grp['Man'] == 100]['total_miti'].values[0]),
        'Combined Sink': subs_data_for_violins2.groupby(other_vars).apply(lambda grp : grp[grp['Man'] == 0][CONSIDERED_C_SINK].values[0] - grp[grp['Man'] == 100][CONSIDERED_C_SINK].values[0]),
        'Carbon Sink A': subs_data_for_violins2.groupby(other_vars).apply(lambda grp : grp[grp['Man'] == 0][CONSIDERED_C_SINK].values[0]),
        'Carbon Sink B': subs_data_for_violins2.groupby(other_vars).apply(lambda grp : grp[grp['Man'] == 100][CONSIDERED_C_SINK].values[0]),
        'Miti A': subs_data_for_violins2.groupby(other_vars).apply(lambda grp : grp[grp['Man'] == 0]['total_miti'].values[0]),
        'Miti B': subs_data_for_violins2.groupby(other_vars).apply(lambda grp : grp[grp['Man'] == 100]['total_miti'].values[0]),
    }).melt(value_vars=['Total Mitigation', 'Combined Sink'], id_vars=['Carbon Sink A', 'Carbon Sink B', 'Miti A', 'Miti B'], var_name='Mitigation Type', ignore_index=False)
    diffs_unmanaged['variable'] = 'stop harvesting'

    all_diffs.insert(-1, diffs_unmanaged)

    all_diffs.append(diffs_material_residence_increase)

    plt.style.use('seaborn')
    diffs_as_df = pd.concat(all_diffs)

    carbon_sink_miti = diffs_as_df[diffs_as_df['Mitigation Type'] == 'Combined Sink']
    assert np.max(np.abs(carbon_sink_miti['value']-(carbon_sink_miti['Carbon Sink A']-carbon_sink_miti['Carbon Sink B']))) < 0.00000001

    sns.violinplot(ax=ax, data=diffs_as_df, x='variable', y='value', hue='Mitigation Type', split=True, inner='quartile', linewidth=1, bw=None, scale='width')

    if forest_type:
        if forest_type == 'bd':
            ticks = [all_diff['variable'].values[0] for all_diff in all_diffs]
            print(ticks)
            ax.set_xticklabels(ticks, rotation = -45, ha='left', fontsize=9)
        else:
            ax.set_xticks([])

        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.get_legend().remove()
        ax.text(1.2, ymax*0.9, forest_age_name[disturb] + ' ' + forest_type, ha='center', va='center', fontsize=12, fontweight='bold', alpha=0.5)
    else:

        plt.xticks(rotation = -45, ha='left', fontsize=13)
        plt.legend(loc='lower left')
        plt.xlabel(None)
        plt.ylabel('$\Delta$ Mitigation potential (kgC/m$^2$)', fontsize=13)

    if ymin and ymax:
        ax.set_ylim([ymin, ymax])

    fig.suptitle('Mitigation impacts until '+str(end_year))

    if letter:
        plt.text(-0.05, 1.05, letter, ha='center', va='center', fontsize=15, fontweight='bold', alpha=1.0, transform=ax.transAxes)

    return diffs_as_df, all_diffs
