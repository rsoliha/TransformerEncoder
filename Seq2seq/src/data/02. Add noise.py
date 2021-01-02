
def alter_scale_data(x_divide_by, y_divide_by):
    import pandas as pd
    import numpy as np

    Orig_data = pd.read_csv("../../data/processed/Orig_data_rico.csv")
    Orig_data = Orig_data.drop(["Unnamed: 0"], axis=1)

    #Save Uniq
    uniq = Orig_data['label'].unique()
    uniqe = pd.DataFrame(columns=["old_labels"])
    uniqe["old_labels"] = uniq
    #uniqe.to_csv("../../data/processed/unique_labels_nishit.csv")

    #Orig_data['xmin']=Orig_data['xmin']/10
    #Orig_data['xmax'] = Orig_data['xmax'] / 10
    #Orig_data['ymin'] = Orig_data['ymin'] / 10
    #Orig_data['ymax'] = Orig_data['ymax'] / 10

    label_map = pd.read_csv('../../data/processed/unique_labels.csv', sep=';')
    #Orig_data = Orig_data.merge(label_map, how='left', left_on='label', right_on='old_labels')
    #Orig_data = Orig_data.drop(['old_labels', 'label'], axis=1, errors='ignore')
    Orig_data.rename(columns={'label':'new_labels'}, inplace=True)
    Orig_data=Orig_data[~(Orig_data['new_labels']=='ignore')]

    #Removing 0 width and 0 height elements
    Orig_data=Orig_data[~(Orig_data['xmin']==Orig_data['xmax'])]
    Orig_data = Orig_data[~(Orig_data['ymin'] == Orig_data['ymax'])]
    Orig_data.to_csv("../../data/processed/data_scaled_labelled_to_use.csv")

    Orig_data['xmin_int']=Orig_data['xmin']/x_divide_by
    Orig_data['xmax_int'] = Orig_data['xmax'] / x_divide_by
    Orig_data['ymin_int'] = Orig_data['ymin'] / y_divide_by
    Orig_data['ymax_int'] = Orig_data['ymax'] / y_divide_by

    #discretizing output
    Orig_data['xmin_int']=Orig_data['xmin_int'].apply(np.floor)
    Orig_data['ymin_int'] = Orig_data['ymin_int'].apply(np.floor)
    Orig_data['xmax_int'] = Orig_data['xmax_int'].apply(np.floor)
    Orig_data['ymax_int'] = Orig_data['ymax_int'].apply(np.floor)

    #Removing 0 width and 0 height elements
    Orig_data = Orig_data[~(Orig_data['xmin_int'] == Orig_data['xmax_int'])]
    Orig_data = Orig_data[~(Orig_data['ymin_int'] == Orig_data['ymax_int'])]

    #Adding
    Orig_data = Orig_data.reset_index()
    Orig_data.rename(columns={'index':'eltID'}, inplace=True)
    Orig_data['width']=Orig_data['xmax_int']-Orig_data['xmin_int']
    Orig_data['height']=Orig_data['ymax_int']-Orig_data['ymin_int']

    #zero_y=np.zeros(Orig_data['ymax_int'].max()+1)
    #Orig_data['vector_height']=zero_y
    #Orig_data['vector_height'][Orig_data['height']]=1

    return Orig_data, Orig_data['xmax_int'].max(), Orig_data['ymax_int'].max()

def add_noise(Orig_data, xmax, ymax):
    import numpy as np

    noise_xmin = np.round(np.random.normal(loc=0, scale=0.5, size=Orig_data.shape[0]))
    noise_ymin = np.round(np.random.normal(loc=0, scale=0.5, size=Orig_data.shape[0]))
    noise_ymax = np.round(np.random.normal(loc=0, scale=0.5, size=Orig_data.shape[0]))
    noise_xmax = np.round(np.random.normal(loc=0, scale=0.5, size=Orig_data.shape[0]))

    Orig_data['xmin_noisy'] = Orig_data['xmin_int']+noise_xmin
    Orig_data['ymin_noisy'] = Orig_data['ymin_int'] + noise_ymin
    Orig_data['ymax_noisy'] = Orig_data['ymax_int'] + noise_ymax
    Orig_data['xmax_noisy'] = Orig_data['xmax_int'] + noise_xmax

    Orig_data.loc[Orig_data['xmin_noisy'] < 0, 'xmin_noisy'] = 0
    Orig_data.loc[Orig_data['ymin_noisy'] < 0, 'ymin_noisy'] = 0
    Orig_data.loc[Orig_data['ymax_noisy'] > ymax, 'ymax_noisy'] = ymax
    Orig_data.loc[Orig_data['xmax_noisy'] > xmax, 'xmax_noisy'] = xmax

    Orig_data['width_noisy'] = Orig_data['xmax_noisy'] - Orig_data['xmin_noisy']
    Orig_data['height_noisy'] = Orig_data['ymax_noisy'] - Orig_data['ymin_noisy']

    Orig_data.drop(["ymax_int","xmax_int", "ymax_noisy", "xmax_noisy"], axis=1, inplace=True)
    Orig_data.drop_duplicates(subset=["ymin_int","xmin_int","UIID","new_labels","width","height"], inplace=True)

    Orig_data = Orig_data.sort_values(['UIID', 'ymin_noisy', 'xmin_noisy']).reset_index()
    Orig_data = Orig_data.drop(columns=['index'])
    return Orig_data

def data_format(data):
    #To include prefix
    data['xmin_int']='xm'+data['xmin_int'].astype(int).astype(str)
    data['ymin_int'] = 'ym' + data['ymin_int'].astype(int).astype(str)
    data['height'] = 'ht' + data['height'].astype(int).astype(str)
    data['width'] = 'wd' + data['width'].astype(int).astype(str)

    data['xmin_noisy'] = 'xm' + data['xmin_noisy'].astype(int).astype(str)
    data['ymin_noisy'] = 'ym' + data['ymin_noisy'].astype(int).astype(str)
    data['height_noisy'] = 'ht' + data['height_noisy'].astype(int).astype(str)
    data['width_noisy'] = 'wd' + data['width_noisy'].astype(int).astype(str)

    #Including labels
    data['seq_x'] = data['new_labels'].astype(str)+' '+data['xmin_noisy'].astype(str)+' '+data['ymin_noisy'].astype(str)+' '+data['height_noisy'].astype(str)+' '+data['width_noisy'].astype(str)
    data['seq_y'] = data['new_labels'].astype(str) + ' ' + data['xmin_int'].astype(str) + ' ' + data['ymin_int'].astype(str) + ' ' + data['height'].astype(str) + ' ' + data['width'].astype(str)

    #Excluding labels with prefix
    #data['seq_x'] = data['xmin_noisy'].astype(str) + ' ' + data['ymin_noisy'].astype(str) + ' ' + data['height_noisy'].astype(str) + ' ' + data['width_noisy'].astype(str)
    #data['seq_y'] = data['xmin_int'].astype(str) + ' ' + data['ymin_int'].astype(str) + ' ' + data['height'].astype(str) + ' ' + data['width'].astype(str)

    # Excluding labels only int
    #data['seq_x'] = data['xmin_noisy'].astype(int).astype(str) + ' ' + data['ymin_noisy'].astype(int).astype(str) + ' ' + data['height_noisy'].astype(int).astype(str) + ' ' + data['width_noisy'].astype(int).astype(str)
    #data['seq_y'] = data['xmin_int'].astype(int).astype(str) + ' ' + data['ymin_int'].astype(int).astype(str) + ' ' + data['height'].astype(int).astype(str) + ' ' + data['width'].astype(int).astype(str)

    fdata_x = data.groupby(['UIID'])['seq_x'].apply(' '.join).reset_index()
    fdata_y = data.groupby(['UIID'])['seq_y'].apply(' '.join).reset_index()

    fdata= fdata_x.merge(fdata_y, how="inner", on="UIID")
    fdata.drop_duplicates(subset=["seq_y"], inplace=True)
    return fdata


def main():
    data, xmax, ymax=alter_scale_data(x_divide_by=25, y_divide_by=25)
    noisy_data= add_noise(data, xmax, ymax)
    fdata = data_format(noisy_data)
    return data, xmax, ymax, noisy_data, fdata


if __name__ == '__main__':
    data, xmax, ymax, noisy_data, fdata =main()
    fdata.to_csv("../../data/processed/Sequence_data_with_prefix.csv")