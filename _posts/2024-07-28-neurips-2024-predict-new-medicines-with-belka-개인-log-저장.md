---
title: "NeurIPS 2024 - Predict New Medicines with BELKA (개인 log 저장)"
date: 2024-07-28 17:50:19
categories:
  - 프로젝트
---

catboost, lgb, reg, transformer, lstm   
  
probabilities = torch.sigmoid(model(X\_test\_smiles, X\_test\_masks, X\_test\_proteins)).numpy().flatten()   
  
최근 연구는 그래프와 같은 이산 및 구조화된 데이터에 확산을 확장하고 있습니다 [35]. 이러한 연구는 분자 설계 [15, 27, 8] 등의 분야에 적용됩니다.   
  
3072   
  
/kaggle/input/0630-leashbio-data/lightgbm\_model\_bind\_BRD4.txt   
  
/kaggle/input/belka-shrunken-train-set/test.parquet   
  
"belka-shrunken-train-set"   
  
나는 내가 사용할 모델을 self - attention  다음 cross attention을 사용하길 원하고, residula과 drop, gelu로 바꾸면 좋겠어. 총 블록의 갯수는 4면 좋겠네

temporary-0704

```
temporary-0704

/kaggle/input/leash-BELKA/train.parquet

File: /kaggle/input/leash-BELKA/train.parquet
Shape: (295246830, 7)
Column names:
['id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles', 'protein_name', 'binds']

/kaggle/input/leash-BELKA/test.parquet

File: /kaggle/input/leash-BELKA/test.parquet
Shape: (1674896, 6)
'id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles', 'protein_name'

File: /kaggle/input/belka-enc-dataset/test_enc.parquet
Shape: (1674896, 142)
'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'enc7', 'enc8', 'enc9', 'enc10', 'enc11', 'enc12', 'enc13', 'enc14', 'enc15', 'enc16', 'enc17', 'enc18', 'enc19', 'enc20', 'enc21', 'enc22', 'enc23', 'enc24', 'enc25', 'enc26', 'enc27', 'enc28', 'enc29', 'enc30', 'enc31', 'enc32', 'enc33', 'enc34', 'enc35', 'enc36', 'enc37', 'enc38', 'enc39', 'enc40', 'enc41', 'enc42', 'enc43', 'enc44', 'enc45', 'enc46', 'enc47', 'enc48', 'enc49', 'enc50', 'enc51', 'enc52', 'enc53', 'enc54', 'enc55', 'enc56', 'enc57', 'enc58', 'enc59', 'enc60', 'enc61', 'enc62', 'enc63', 'enc64', 'enc65', 'enc66', 'enc67', 'enc68', 'enc69', 'enc70', 'enc71', 'enc72', 'enc73', 'enc74', 'enc75', 'enc76', 'enc77', 'enc78', 'enc79', 'enc80', 'enc81', 'enc82', 'enc83', 'enc84', 'enc85', 'enc86', 'enc87', 'enc88', 'enc89', 'enc90', 'enc91', 'enc92', 'enc93', 'enc94', 'enc95', 'enc96', 'enc97', 'enc98', 'enc99', 'enc100', 'enc101', 'enc102', 'enc103', 'enc104', 'enc105', 'enc106', 'enc107', 'enc108', 'enc109', 'enc110', 'enc111', 'enc112', 'enc113', 'enc114', 'enc115', 'enc116', 'enc117', 'enc118', 'enc119', 'enc120', 'enc121', 'enc122', 'enc123', 'enc124', 'enc125', 'enc126', 'enc127', 'enc128', 'enc129', 'enc130', 'enc131', 'enc132', 'enc133', 'enc134', 'enc135', 'enc136', 'enc137', 'enc138', 'enc139', 'enc140', 'enc141'

File: /kaggle/input/belka-enc-dataset/train_enc.parquet
Shape: (98415610, 145)
'enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'enc7', 'enc8', 'enc9', 'enc10', 'enc11', 'enc12', 'enc13', 'enc14', 'enc15', 'enc16', 'enc17', 'enc18', 'enc19', 'enc20', 'enc21', 'enc22', 'enc23', 'enc24', 'enc25', 'enc26', 'enc27', 'enc28', 'enc29', 'enc30', 'enc31', 'enc32', 'enc33', 'enc34', 'enc35', 'enc36', 'enc37', 'enc38', 'enc39', 'enc40', 'enc41', 'enc42', 'enc43', 'enc44', 'enc45', 'enc46', 'enc47', 'enc48', 'enc49', 'enc50', 'enc51', 'enc52', 'enc53', 'enc54', 'enc55', 'enc56', 'enc57', 'enc58', 'enc59', 'enc60', 'enc61', 'enc62', 'enc63', 'enc64', 'enc65', 'enc66', 'enc67', 'enc68', 'enc69', 'enc70', 'enc71', 'enc72', 'enc73', 'enc74', 'enc75', 'enc76', 'enc77', 'enc78', 'enc79', 'enc80', 'enc81', 'enc82', 'enc83', 'enc84', 'enc85', 'enc86', 'enc87', 'enc88', 'enc89', 'enc90', 'enc91', 'enc92', 'enc93', 'enc94', 'enc95', 'enc96', 'enc97', 'enc98', 'enc99', 'enc100', 'enc101', 'enc102', 'enc103', 'enc104', 'enc105', 'enc106', 'enc107', 'enc108', 'enc109', 'enc110', 'enc111', 'enc112', 'enc113', 'enc114', 'enc115', 'enc116', 'enc117', 'enc118', 'enc119', 'enc120', 'enc121', 'enc122', 'enc123', 'enc124', 'enc125', 'enc126', 'enc127', 'enc128', 'enc129', 'enc130', 'enc131', 'enc132', 'enc133', 'enc134', 'enc135', 'enc136', 'enc137', 'enc138', 'enc139', 'enc140', 'enc141', 'bind1', 'bind2', 'bind3'

File: /kaggle/input/temporary-0704/test.parquet
Shape: (1674896, 9)
Column names:
['id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles', 'protein_name', 'is_BRD4', 'is_HSA', 'is_sEH']

File: /kaggle/input/temporary-0704/train.parquet
Shape: (98415610, 7)
['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles', 'binds_BRD4', 'binds_HSA', 'binds_sEH']
```

```
File: /kaggle/input/shrunken-train-set/test_fold.parquet
Shape: (878022, 4)

First few rows:
   buildingblock1_smiles  buildingblock2_smiles  buildingblock3_smiles  \
0                      0                     17                     17   
1                      0                     17                     87   
2                      0                     17                     99   
3                      0                     17                    244   
4                      0                     17                    394   

   bb1_scaffold_idx  
0                91  
1                91  
2                91  
3                91  
4                91  

Column names:
['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'bb1_scaffold_idx']

File: /kaggle/input/shrunken-train-set/test.parquet
Shape: (878022, 7)

First few rows:
   buildingblock1_smiles  buildingblock2_smiles  buildingblock3_smiles  \
0                      0                     17                     17   
1                      0                     17                     87   
2                      0                     17                     99   
3                      0                     17                    244   
4                      0                     17                    394   

                                     molecule_smiles  is_BRD4  is_HSA  is_sEH  
0  C#CCCC[C@H](Nc1nc(Nc2ccc(C=C)cc2)nc(Nc2ccc(C=C...     True    True    True  
1  C#CCCC[C@H](Nc1nc(Nc2ccc(C=C)cc2)nc(Nc2ncnc3c2...     True    True    True  
2  C#CCCC[C@H](Nc1nc(NCC2(O)CCCC2(C)C)nc(Nc2ccc(C...     True    True    True  
3  C#CCCC[C@H](Nc1nc(Nc2ccc(C=C)cc2)nc(Nc2sc(Cl)c...     True    True    True  
4  C#CCCC[C@H](Nc1nc(NCC2CCC(SC)CC2)nc(Nc2ccc(C=C...     True    True    True  

Column names:
['buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles', 'is_BRD4', 'is_HSA', 'is_sEH']
```
