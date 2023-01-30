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

# Process for GLoRIA
# df = pd.read_csv('/data4/candid-ptx/Pneumothorax_reports.csv')
# df['Pneumothorax'] = (df.EncodedPixels != '-1').astype(int)

# chest_tube = pd.read_csv('/data4/candid-ptx/chest_tube.csv')
# chest_tube.columns = ['SOPInstanceUID', 'annotation_ref', 'mask_rle']
# chest_tube['Chest Tube'] = 1
# df = df.merge(chest_tube[['SOPInstanceUID', 'Chest Tube']], on='SOPInstanceUID', how='left')
# df['Chest Tube'] = (~df['Chest Tube'].isnull()).astype(int)

# fracture = pd.read_csv('/data4/candid-ptx/Rib_fracture_mask_rle.csv')
# fracture.columns = ['SOPInstanceUID', 'annotation_ref', 'mask_rle']
# fracture['Rib Fracture'] = 1
# df = df.merge(fracture[['SOPInstanceUID', 'Rib Fracture']], on='SOPInstanceUID', how='left')
# df['Rib Fracture'] = (~df['Rib Fracture'].isnull()).astype(int)

# print(df.shape)

# df['Report Length'] = df['Report'].str.len()
# print(f'Average character length of report: ',np.mean(df['Report Length']), 'number of reports with length > 1000: ', df[df['Report Length'] > 1000].shape[0])

# # Match alphanumeric characters and periods
# tokenizer = RegexpTokenizer(r"\w+|\.|:")
# df['Report Impression'] = df['Report'].apply(lambda x: re.sub(r'(\[|\]|\(|\))', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\n(\s|!PERSONALNAME!|PA|AP)*(Chest|CXR)[^(:|\n|\.)]*(:|\n)', 'study:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\n(\s|!PERSONALNAME!|PA|AP)*(CHEST|CXR)[^(:|\n|\.)]*(:|\n)', 'study:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\n\s*!PERSONALNAME!\n', 'study:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\nCOMMENT:\n<###>\n', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\nComment:\nNo change.\n', '', x))
# df['Report Impression'] = df['Report Impression'].str.lower()

# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'(report|findings)(:|\n)+', 'findings:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'(comments|comment|impression)(:|\n)+', 'impression:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'comparison:[^(\.|\n)]*(\.|\n)', 'comparison:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'comparison image:[^(\.|\n)]*(\.|\n)', 'comparison:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'(comparison is made)[^(\.|\n)]*(\.|\n)*', 'comparison:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'(comparison was made)[^(\.|\n)]*(\.|\n)*', 'comparison:', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\ndr.*\n', '\ndoctor:', x))

# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\n', ' ', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'\t', ' ', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r' \.', '.', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r' :', ':', x))

# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?study:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?medical question:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?comparison:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?findings:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?chest:', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'^.*?impression:', '', x))

# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'report in consultation.*', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'reported in consultation.*', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'transcribed by.*', '', x))
# df['Report Impression'] = df['Report Impression'].apply(lambda x: re.sub(r'doctor.*', '', x))

# df['Report Impression'] = df['Report Impression'].str.strip(" ")
# df['Report Impression Length'] = df['Report Impression'].str.len()
# df['Report Impression Num Sentences'] = df['Report Impression'].str.split('.').str.len()

# print(f'Average character length of impression: ',np.mean(df['Report Impression Length']), 'number of impressions with length > 1000: ', df[df['Report Impression Length'] > 1000].shape[0])
# print(np.mean(df['Report Impression Num Sentences']), np.median(df['Report Impression Num Sentences']), np.min(df['Report Impression Num Sentences']), np.max(df['Report Impression Num Sentences']), np.std(df['Report Impression Num Sentences']))

# # we need reports to run GLoRIA - handle this internally
# print("CANDID-PTX before removing NULL reports: ", df.shape, df.StudyInstanceUID.nunique())
# df = df[~df['Report Impression'].isna()]
# df = df[df['Report Impression'] != '']
# print("CANDID-PTX after removing NULL reports: ", df.shape, df.StudyInstanceUID.nunique())
# df = df.drop_duplicates('SOPInstanceUID').reset_index(drop=True)
# print("CANDID-PTX after dedup: ", df.shape, df.StudyInstanceUID.nunique())

# df['Path'] = '/data4/candid-ptx/images/' + df['SOPInstanceUID'] + '.jpg'

# # sample 10% each for val and test
# valid_test_inds = random.sample(range(df.shape[0]), k=df.shape[0] // 5)
# test_inds = random.sample(valid_test_inds, k=len(valid_test_inds) // 2)

# df['Split'] = 'train'
# df.loc[valid_test_inds, 'Split'] = 'valid'
# df.loc[test_inds, 'Split'] = 'test'

# df['Frontal/Lateral'] = 'Frontal'

# print(f"train: {df[df['Split'] == 'train'].shape}, valid: {df[df['Split'] == 'valid'].shape}, test: {df[df['Split'] == 'test'].shape}")

# df = df[['Path', 'Frontal/Lateral', 'Report Impression', 'Pneumothorax', 'Rib Fracture', 'Chest Tube', 'Split']]
# df.to_csv('/data4/candid-ptx/candid_ptx.csv', index=False)

# # NOTE: only use train for pre-training training and validation!
# train_val_df = df[df.Split.isin(['train', 'valid'])]
# train_val_df.to_csv('/data4/candid-ptx/candid_ptx_train_val.csv', index=False)

# test_df = df[df.Split.isin(['test'])]
# test_df.to_csv('/data4/candid-ptx/candid_ptx_test.csv', index=False)


# now process for MoCo
df = pd.read_csv('/data4/candid-ptx/candid_ptx_train_val.csv')
df['file_path'] = df.Path
df = df[df[CHEXPERT_VIEW_COL] == 'Frontal']
print(f'CANDID-PTX MoCo: {df.shape[0]} images')
df.to_csv('/data4/candid-ptx/candid_ptx_moco.csv', index=False)