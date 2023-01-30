import os 
import numpy as np
import pandas as pd
import re
import random
from nltk.tokenize import RegexpTokenizer

random.seed(0)

CHEXPERT_VIEW_COL = "Frontal/Lateral"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_SPLIT_COL = "Split"


# # # process to get all views, etc
# # train_df = pd.read_csv('/data4/intermountain/chexed_v2/data/train_2009-2021__val_2021__test_2021/train.csv')
# # val_df = pd.read_csv('/data4/intermountain/chexed_v2/data/train_2009-2021__val_2021__test_2021/valid.csv')
# # test_df = pd.read_csv('/data4/intermountain/chexed_v2/data/train_2009-2021__val_2021__test_2021/test.csv')

# # df = pd.concat([train_df, val_df, test_df])
# # df = df.drop('Unnamed: 0', axis=1)

# # # add in new reports
# # print(f"Before adding in new reports: {sum(~df['Report'].isna())}/{df.shape[0]}")
# # new_reports = pd.read_csv('/data4/intermountain/chexed_v2/data/intermountain_rad_reports.csv')
# # new_reports.columns = ['AccessionNumber', 'ExamDesc', 'NewReport']
# # df = df.merge(new_reports[['AccessionNumber', 'NewReport']], on='AccessionNumber', how='left')
# # df['Report'] = df['Report'].fillna(df['NewReport'])
# # df = df.drop('NewReport', axis=1)
# # print(f"After adding in new reports: {sum(~df['Report'].isna())}/{df.shape[0]}")

# # # had to move data to /data4
# # df['Path'] = df.Path.replace({
# #     '/home/cvanuden/chexed/': '/data4/intermountain/chexed_v2/'
# # }, regex=True)

# # # go back and get all views - had deduped here
# # df['StudyDir'] = df.Path.str.split('/').str[:-1].str.join('/')

# # all_views_data = []
# # for study_dir in df['StudyDir'].unique().tolist():
# #     for view in os.listdir(study_dir):
# #         path = os.path.join(study_dir, view)
# #         row = df[df.StudyDir == study_dir]
# #         row['Path'] = path
# #         all_views_data.append(row)
# # df = pd.concat(all_views_data)

# # # check if two views
# # # first view is always frontal
# # # second is lateral
# # df['TempViewNum'] = df.Path.str.split('/').str[-1].str.contains('1')
# # df['TempReport'] = df.Report.str.lower()
# # df['TempViewReport'] = df.TempReport.str.contains('1 view|one view')

# # df[CHEXPERT_VIEW_COL] = df.apply(lambda row: row['TempViewNum'] or row['TempViewReport'], axis=1)
# # df[CHEXPERT_VIEW_COL] = df[CHEXPERT_VIEW_COL].replace({
# #     True: 'Frontal',
# #     False: 'Lateral',
# # })

# # # otherwise, frontal/lateral not applicable
# # df['NumViews'] = df['StudyDir'].apply(lambda study_dir: len(df[df['StudyDir'] == study_dir]))
# # # df.loc[df['NumViews'] != 2, CHEXPERT_VIEW_COL] = None

# # # make train/val/test single column
# # df[CHEXPERT_SPLIT_COL]=df[['train', 'valid', 'test']].values.argmax(1).astype(int)
# # df[CHEXPERT_SPLIT_COL] = df[CHEXPERT_SPLIT_COL].replace({
# #     0: 'train',
# #     1: 'valid',
# #     2: 'test',
# # })

# # df['PathExists'] = df.Path.apply(os.path.exists)

# # df = df[[
# #     'AccessionNumber',
# #     'StudyDir',
# #     'Path',
# #     'PathExists',
# #     'Frontal/Lateral',
# #     'NumViews',
# #     'Pneumonia',
# #     'Effusion',
# #     'NumLobes',
# #     'FeedingTube',
# #     'Covid',
# #     'Report',
# #     'DataDate',
# #     'DataLoc',
# #     'DataSource',
# #     'Split'
# # ]]

# # print("Intermountain Master: ", df.shape)
# # df.to_csv('/data4/intermountain/chexed_v2/data/intermountain_all_views_master.csv', index=False)

# now process for GLoRIA
# df = pd.read_csv('/data4/intermountain/chexed_v2/data/intermountain_all_views_master.csv')

# df['Report Length'] = df['Report'].str.len()
# print(f'Average character length of report: ',np.mean(df['Report Length']), 'number of reports with length > 1000: ', df[df['Report Length'] > 1000].shape[0])

# # Match alphanumeric characters and periods
# tokenizer = RegexpTokenizer(r"\w+|\.|:")
# # tokenizer = RegexpTokenizer(r"\w+|\.")
# df['Report Impression'] = df['Report'].str.lower()
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\n', ' ', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\t', ' ', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))

# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r' \.', '.', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r' :', ':', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?conclusion:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?impression:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?impressions:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?pression:', '', x))

# # df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?finding:', '', x))
# # df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?findings:', '', x))
# # df['Report Impression'] = df['Report Impression'].apply(lambda x: '.'.join(word.strip(' ').replace('conclusion:', 'IMPRESSION:').replace('impressions:', 'IMPRESSION:').replace('in pression:', 'IMPRESSION:').replace('impression:', 'IMPRESSION:').replace('pression:', 'IMPRESSION:') for word in x.split('.')))

# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'this report was electronically signed by.*', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'this report was dictated by.*', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'signed by.*', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'dictated by.*', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'radiologist:.*', '', x))

# df['Report Impression'] = df['Report Impression'].str.strip(" ")
# df['Report Impression Length'] = df['Report Impression'].str.len()
# df['Report Impression Num Sentences'] = df['Report Impression'].str.split('.').str.len()

# print(f'Average character length of impression: ',np.mean(df['Report Impression Length']), 'number of impressions with length > 1000: ', df[df['Report Impression Length'] > 1000].shape[0])
# print(np.mean(df['Report Impression Num Sentences']), np.median(df['Report Impression Num Sentences']), np.min(df['Report Impression Num Sentences']), np.max(df['Report Impression Num Sentences']), np.std(df['Report Impression Num Sentences']))

# # we need reports to run GLoRIA - handle this internally
# print("Intermountain GLoRIA before removing NULL reports: ", df.shape, df.StudyDir.nunique())
# df = df[~df['Report Impression'].isna()]
# print("Intermountain GLoRIA after removing NULL reports: ", df.shape, df.StudyDir.nunique())

# print("Intermountain Supervised num frontal images: ", df[df['Frontal/Lateral'] == 'Frontal'].shape, df[df['Frontal/Lateral'] == 'Frontal'].StudyDir.nunique())

# n_studies = df.StudyDir.nunique()
# print("Intermountain GLoRIA: ", df.shape, n_studies)

# df = df.reset_index(drop=True)
# no_pneumonia_inds = df[df.Pneumonia == 0].index

# df['Pneumonia+Effusion'] = df['Effusion']
# df.loc[no_pneumonia_inds, 'Pneumonia+Effusion'] = 0

# df['Pneumonia+NumLobes'] = df['NumLobes']
# df.loc[no_pneumonia_inds, 'Pneumonia+NumLobes'] = 0


# df = df[['AccessionNumber', 'Path', 'Frontal/Lateral', 'Pneumonia', 'Effusion', 'NumLobes', 'Pneumonia+Effusion', 'Pneumonia+NumLobes', 'Report Impression', 'Split']]
# df.to_csv('/data4/intermountain/chexed_v2/data/intermountain.csv', index=False)

# # NOTE: only use train for pre-training training and validation!
# train_val_df = df[df.Split.isin(['train', 'valid'])]
# train_val_df.to_csv('/data4/intermountain/chexed_v2/data/intermountain_train_val.csv', index=False)

# test_df = df[df.Split.isin(['test'])]
# test_df.to_csv('/data4/intermountain/chexed_v2/data/intermountain_test.csv', index=False)

# now process for MoCo
df = pd.read_csv('/data4/intermountain/chexed_v2/data/intermountain_train_val.csv')
df['file_path'] = df.Path
df = df[df[CHEXPERT_VIEW_COL] == 'Frontal']
print(f'Intermountain MoCo: {df.shape[0]} images, {df.AccessionNumber.nunique()} studies')
df.to_csv('/data4/intermountain/chexed_v2/data/intermountain_moco.csv', index=False)

# df = pd.read_csv('/home/cvanuden/git-repos/MedAug/moco/processed_pretrain_w_lat_count.csv')
# print(df.shape)
# df = df[df.laterality == 'frontal.jpg']
# print(df.shape)

# df['patient_study'] = df['patient'] + '_' + df['study']
# patient_studies = set(df['patient_study'])
# sampled_patient_studies = random.sample(patient_studies, k=n_studies)
# df = df[df.patient_study.isin(sampled_patient_studies)]

# print(f'CheXpert-Small MoCo: {df.shape[0]} images, {df.patient_study.nunique()} studies')
# df.to_csv('/data4/intermountain/chexed_v2/data/chexpert_small_moco.csv', index=False)

# now process for MedAug
# medaug_df = pd.read_csv('/home/cvanuden/git-repos/MedAug/moco/processed_pretrain_w_lat_count.csv')
# columns = medaug_df.columns.tolist()

# df = pd.read_csv('/data4/intermountain/chexed_v2/data/intermountain_all_views_master.csv')
# df = df[df.Split == 'train']

# print("Intermountain MedAug before removing single-view and outlier studies: ", df.shape, df.StudyDir.nunique())
# df = df[df.NumViews > 1]
# df = df[df.NumViews < 127]
# print("Intermountain MedAug after removing single-view and outlier studies: ", df.shape, df.StudyDir.nunique())

# df = df.merge(df[['StudyDir', 'Path']], on='StudyDir')
# df = df[df.Path_x != df.Path_y]

# # df = df[df['Frontal/Lateral'] == 'Frontal']

# medaug_data = []
# for i, row in df.iterrows():
#     temp_split_path = row['Path_x'].split('/')

#     # base this row off of Path_x
#     patient = temp_split_path[-3] if 'patient' in row['Path_x'] else temp_split_path[-2]
#     study = temp_split_path[-2]
#     laterality = temp_split_path[-1]
#     opposite_laterality = row['Path_y'].split('/')[-1]
#     # same_laterality = 1
#     # diff_laterality = int('2' in laterality)
#     disease = 'no_sym'
#     path_exists = os.path.exists(row['Path_x'])

#     medaug_data.append([row['StudyDir'], row['Path_x'], patient, study, laterality, opposite_laterality, disease, path_exists])

# df = pd.DataFrame(medaug_data, columns=['StudyDir', 'file_path', 'patient', 'study', 'laterality', 'opposite laterality', 'disease', 'path_exists'])

# print("Intermountain MedAug: ", df.shape, df.StudyDir.nunique())
# n_medaug_studies = df.StudyDir.nunique()
# df.to_csv('/data4/intermountain/chexed_v2/data/intermountain_medaug.csv', index=False)

# # now, process CheXpert MedAug to get "small" version that matches number of studies in Intermountain
# medaug_df = pd.read_csv('/home/cvanuden/git-repos/MedAug/moco/processed_pretrain_w_lat_count.csv')
# medaug_df['patient_study'] = medaug_df['patient'] + '_' + medaug_df['study']
# patient_studies = set(medaug_df['patient_study'])
# sampled_patient_studies = random.sample(patient_studies, k=n_medaug_studies)
# medaug_df = medaug_df[medaug_df.patient_study.isin(sampled_patient_studies)]
# print("CheXpert-Small MedAug: ", medaug_df.shape, medaug_df.patient_study.nunique())
# medaug_df.to_csv('/home/cvanuden/git-repos/MedAug/moco/processed_pretrain_w_lat_count_chexpert_small.csv', index=False)

# # now only get studies that overlap
# gloria_df = pd.read_csv('/data4/intermountain/chexed_v2/data/intermountain_gloria.csv')
# medaug_df = pd.read_csv('/data4/intermountain/chexed_v2/data/intermountain_medaug.csv')

# print(gloria_df.shape, medaug_df.shape, gloria_df.StudyDir.nunique(), medaug_df.StudyDir.nunique())
# gloria_df = gloria_df[gloria_df.StudyDir.isin(medaug_df.StudyDir.unique())]
# medaug_df = medaug_df[medaug_df.StudyDir.isin(gloria_df.StudyDir.unique())]
# print(gloria_df.shape, medaug_df.shape, gloria_df.StudyDir.nunique(), medaug_df.StudyDir.nunique())

# gloria_df.to_csv('/data4/intermountain/chexed_v2/data/intermountain_gloria.csv', index=False)
# medaug_df.to_csv('/data4/intermountain/chexed_v2/data/intermountain_medaug.csv', index=False)
