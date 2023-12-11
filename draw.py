# -*- coding: UTF-8 -*-
import re
import numpy as np
import matplotlib.pyplot as plt

""" lobe """
lobe_name = ['All', 'Frontal Lobe', 'Temporal Lobe', 'Parietal Lobe', 
             'Insular Lobe', 'Limbic Lobe', 'Occipital Lobe', 'Subcortical Nuclei']
lobe_auc = [[0.9705882352941175, 0.9626517273576096, 0.9486461251167133],
            [0.9295051353874884, 0.9197012138188608, 0.9710550887021476],
            [0.9117647058823529, 0.8314659197012139, 0.8445378151260503],
            [0.8748832866479924, 0.8291316526610644, 0.8632119514472456],
            [0.8949579831932771, 0.8632119514472456, 0.8968253968253967],
            [0.9126984126984128, 0.9687208216619981, 0.9607843137254901], 
            [0.8487394957983193, 0.9215686274509803, 0.7675070028011205],
            [0.8697478991596639, 0.9477124183006536, 0.8902894491129786]]
lobe_logloss = [[0.3361760814676079 , 0.34869968432606774, 0.35855873308430675],
                [0.3129801134555635 , 0.3181792561908074 , 0.2969358745195375],
                [0.39561239441507773, 0.5004270381461514 , 0.5117715748266591],
                [0.5478438350640282 , 0.49200533095165466, 0.5239538148365399],
                [0.4521059868948521 , 0.4551762282587244 , 0.35935263275227064],
                [0.3786424321056641 , 0.3080566487336423 , 0.3146869558795491],
                [0.5384760521367581 , 0.35587533899984763, 0.5048600722191904],
                [0.43384391517388354, 0.3182188387842193 , 0.4829665646186517]]
lobe_auc = [sum(row)/len(row) for row in lobe_auc]
lobe_logloss = [sum(row)/len(row) for row in lobe_logloss]
assert len(lobe_name) == len(lobe_auc) == len(lobe_logloss)

plt.plot(lobe_name, lobe_auc, label='AUC', marker='o')
for i, value in enumerate(lobe_auc):
    plt.text(i, value+0.01, str(round(value,3)), ha='center', va='bottom')
plt.plot(lobe_name, lobe_logloss, label='LogLoss', marker='*')
for i, value in enumerate(lobe_logloss):
    plt.text(i, value+0.01, str(round(value,3)), ha='center', va='bottom')
plt.title('AUC & LogLoss (7 lobes)', fontname='Times New Roman', fontsize=20)
plt.xticks(lobe_name, fontname='Times New Roman', fontsize=16)
plt.grid()
plt.legend()
plt.show()

# lobe: All	AUC=0.9705882352941175	LogLoss=0.3361760814676079
lobe_one_res = '''tensor(0.9035)	tensor(0.)
tensor(0.0865)	tensor(0.)
tensor(0.0871)	tensor(0.)
tensor(0.9038)	tensor(1.)
tensor(0.8981)	tensor(0.)
tensor(0.9033)	tensor(1.)
tensor(0.0949)	tensor(0.)
tensor(0.9064)	tensor(1.)
tensor(0.9023)	tensor(1.)
tensor(0.8927)	tensor(0.)
tensor(0.8743)	tensor(0.)
tensor(0.9051)	tensor(1.)
tensor(0.0868)	tensor(0.)
tensor(0.0867)	tensor(0.)
tensor(0.0865)	tensor(0.)
tensor(0.9065)	tensor(1.)
tensor(0.9043)	tensor(1.)
tensor(0.0867)	tensor(0.)
tensor(0.9039)	tensor(1.)
tensor(0.0885)	tensor(0.)
tensor(0.9055)	tensor(1.)
tensor(0.9037)	tensor(1.)
tensor(0.9046)	tensor(1.)
tensor(0.9063)	tensor(1.)
tensor(0.9056)	tensor(0.)
tensor(0.8728)	tensor(0.)
tensor(0.9063)	tensor(1.)
tensor(0.8987)	tensor(1.)
tensor(0.8993)	tensor(1.)
tensor(0.0868)	tensor(0.)
tensor(0.0923)	tensor(0.)
tensor(0.9017)	tensor(1.)
tensor(0.0867)	tensor(0.)
tensor(0.0874)	tensor(0.)
tensor(0.9058)	tensor(1.)
tensor(0.0908)	tensor(0.)
tensor(0.0894)	tensor(0.)
tensor(0.8964)	tensor(0.)
tensor(0.9062)	tensor(1.)
tensor(0.9064)	tensor(1.)
tensor(0.8933)	tensor(1.)
tensor(0.9063)	tensor(1.)
tensor(0.9012)	tensor(1.)
tensor(0.9016)	tensor(1.)
tensor(0.9049)	tensor(1.)
tensor(0.8987)	tensor(0.)
tensor(0.9052)	tensor(1.)
tensor(0.9058)	tensor(1.)
tensor(0.0871)	tensor(0.)
tensor(0.9058)	tensor(1.)
tensor(0.0880)	tensor(0.)
tensor(0.9050)	tensor(1.)
tensor(0.9055)	tensor(1.)
tensor(0.0871)	tensor(0.)
tensor(0.0866)	tensor(0.)
tensor(0.9064)	tensor(1.)
tensor(0.9054)	tensor(1.)
tensor(0.9064)	tensor(1.)
tensor(0.9034)	tensor(1.)
tensor(0.9043)	tensor(1.)
tensor(0.0865)	tensor(0.)
tensor(0.9061)	tensor(1.)
tensor(0.9060)	tensor(1.)
tensor(0.0868)	tensor(0.)
tensor(0.0871)	tensor(0.)
tensor(0.9004)	tensor(1.)
tensor(0.0869)	tensor(0.)
tensor(0.0866)	tensor(0.)
tensor(0.9050)	tensor(1.)
tensor(0.9060)	tensor(1.)
tensor(0.0864)	tensor(0.)
tensor(0.0872)	tensor(0.)
tensor(0.9048)	tensor(1.)
tensor(0.9038)	tensor(1.)
tensor(0.9037)	tensor(1.)
tensor(0.9054)	tensor(1.)
tensor(0.9064)	tensor(1.)
tensor(0.9008)	tensor(1.)
tensor(0.4508)	tensor(1.)
tensor(0.9034)	tensor(1.)
tensor(0.0868)	tensor(0.)
tensor(0.0866)	tensor(0.)
tensor(0.9003)	tensor(0.)
tensor(0.8986)	tensor(0.)
tensor(0.9048)	tensor(1.)
tensor(0.9039)	tensor(1.)
tensor(0.0876)	tensor(0.)
tensor(0.0870)	tensor(0.)
tensor(0.0868)	tensor(0.)
tensor(0.0867)	tensor(0.)
tensor(0.0869)	tensor(0.)
tensor(0.9061)	tensor(1.)
tensor(0.9029)	tensor(1.)'''
numbers = re.findall(r'\d+\.\d*', lobe_one_res)
pred = [float(numbers[i]) for i in range(len(numbers)) if i % 2 == 0]
true = [int(float(numbers[i])) for i in range(len(numbers)) if not i % 2 == 0]
assert len(pred) == len(true)


""" gyrus """
# gyrus_name = ['All', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus', 'Inferior Frontal Gyrus', 'Orbital Gyrus', 'Precentral Gyrus',
#               'Paracentral Lobule', 'Superior Temporal Gyrus', 'Middle Temporal Gyrus', 'Inferior Temporal Gyrus', 'Fusiform Gyrus',
#               'Parahippocampal Gyrus', 'posterior Superior Temporal Sulcus', 'Superior Parietal Lobule', 'Inferior Parietal Lobule',
#               'Precuneus', 'Postcentral Gyrus', 'Insular Gyrus', 'Cingulate Gyrus', 'MedioVentral Occipital Cortex', 
#               'lateral Occipital Cortex', 'Amygdala', 'Hippocampus', 'Basal Ganglia', 'Thalamus']
gyrus_name = ['All', 'SFG', 'MFG', 'IFG', 'OrG', 'PrG','PCL',
               'STG', 'MTG', 'ITG', 'FuG','PhG', 'pSTS', 
               'SPL', 'IPL','Pcun', 'PoG', 
               'INS', 
               'CG', 
               'MVOcC', 'LOcC', 
               'Amyg', 'Hipp', 'BG', 'Tha']
gyrus_auc = [[0.9719887955182072, 0.9869281045751633, 0.9719887955182072],
             [0.9313725490196078, 0.9206349206349206, 0.9519140989729225],
             [0.8356676003734828, 0.9108309990662931, 0.9379084967320261],
             [0.8940242763772176, 0.7791783380018673, 0.8440709617180205],
             [0.9262371615312791, 0.927170868347339 , 0.9253034547152194],
             [0.8160597572362278, 0.8076563958916899, 0.7838468720821662],
             [0.896358543417367 , 0.8818860877684408, 0.908029878618114 ],
             [0.9477124183006536, 0.9183006535947712, 0.9528478057889822],
             [0.8594771241830065, 0.8258636788048552, 0.8748832866479925],
             [0.9663865546218487, 0.9542483660130718, 0.9822595704948646],
             [0.8893557422969187, 0.8123249299719888, 0.9136321195144723],
             [0.8814192343604108, 0.9112978524743232, 0.8954248366013071],
             [0.945845004668534 , 0.9631185807656396, 0.9715219421101775],
             [0.8655462184873949, 0.8328664799253034, 0.8436041083099907],
             [0.8272642390289449, 0.8739495798319328, 0.9290382819794584],
             [0.9243697478991597, 0.907563025210084 , 0.9206349206349207],
             [0.699813258636788 , 0.6568627450980392, 0.8333333333333333],
             [0.9234360410830997, 0.942110177404295 , 0.9126984126984127],
             [0.9547152194211017, 0.9500466853408029, 0.9140989729225023],
             [0.8099906629318394, 0.746031746031746 , 0.9010270774976656],
             [0.9014939309056955, 0.8281979458450046, 0.9178338001867412],
             [0.9640522875816993, 0.9864612511671336, 0.9817927170868347],
             [0.9187675070028012, 0.8557422969187676, 0.9715219421101775],
             [0.903828197945845 , 0.9411764705882353, 0.9066293183940243],
             [0.9108309990662933, 0.9075630252100839, 0.8632119514472456] ]
gyrus_logloss = [[0.3774568150507361 , 0.364157707112194  , 0.3557573810209777 ],
                 [0.35641000370436227, 0.4387082121543099 , 0.34269746620515984],
                 [0.409206209779479  , 0.35103885495338466, 0.35369441380778327],
                 [0.3584008252091755 , 0.36476627337494427, 0.35780519203371075],
                 [0.39994568925406604, 0.4064223906594925 , 0.40708738654393023],
                 [0.3709179724282271 , 0.3939653873730284 , 0.4417391243454662 ],
                 [0.5733255667532833 , 0.6333083538001606 , 0.586204839611721  ],
                 [0.27907755687412555, 0.40647827490230826, 0.32359113334224476],
                 [0.376753921521312  , 0.4385845348296728 , 0.3580530492286875 ],
                 [0.32138021727191757, 0.39352877031072236, 0.22585282206108434],
                 [0.38632832079410784, 0.41784797276125724, 0.38385883806085375],
                 [0.4452729877117041 , 0.3849089769696607 , 0.4301615939423127 ],
                 [0.37068465744240137, 0.32745084416815795, 0.33594138336680196],
                 [0.43247118947325647, 0.47532718542942426, 0.5694909292927766 ],
                 [0.46440942277137043, 0.4301231244324399 , 0.37466104485676427],
                 [0.4604867791661369 , 0.4802081290096029 , 0.4458948143528397 ],
                 [0.5726964390556203 , 0.5963472852342013 , 0.5278454820358024 ],
                 [0.3776411889909457 , 0.380201783668372  , 0.38555440538910685],
                 [0.4194559007895425 , 0.42221918831614125, 0.5037163935585248 ],
                 [0.5753047883581125 , 0.5011434518322165 , 0.3430815873887223 ],
                 [0.3601178345332771 , 0.3818329795897665 , 0.3418228972230644 ],
                 [0.2949693617759361 , 0.2473291464539728 , 0.2880423543054257 ],
                 [0.3651515853068269 , 0.3763668066731855 , 0.3363056327471282 ],
                 [0.3689283083630569 , 0.34176944614580346, 0.43589367886649105],
                 [0.4486134269595081 , 0.46181930495037327, 0.48333205507457633] ]
gyrus_auc = [sum(row)/len(row) for row in gyrus_auc]
gyrus_logloss = [sum(row)/len(row) for row in gyrus_logloss]
print(len(gyrus_name) , len(gyrus_auc), len(gyrus_logloss))
assert len(gyrus_name) == len(gyrus_auc) == len(gyrus_logloss)

plt.plot(gyrus_name, gyrus_auc, label='AUC', marker='o')
for i, value in enumerate(gyrus_auc):
    plt.text(i, value+0.01, str(round(value,3)), ha='center', va='bottom')
plt.plot(gyrus_name, gyrus_logloss, label='LogLoss', marker='*')
for i, value in enumerate(gyrus_logloss):
    plt.text(i, value+0.01, str(round(value,3)), ha='center', va='bottom')
plt.title('AUC & LogLoss (24 gyrus)', fontname='Times New Roman', fontsize=20)
plt.xticks(gyrus_name, rotation=-45, fontname='Times New Roman', fontsize=16)
plt.grid()
plt.legend()
plt.show()


""" 超参数敏感性实验"""
# epochs = 100; learning_rete vary
lr_list = ['0.0001', '0.0002', '0.0005', '0.001', '0.002', '0.005', '0.01', '0.02', '0.05']
# [lobe],[gyrus]
auc_of_diff_lr = [[0.9691876750700279, 0.9505135387488328, 0.9733893557422968], [0.9859943977591036, 0.9836601307189542 , 0.9747899159663865],
                  [0.9654528478057889, 0.9598506069094304, 0.9407096171802054], [0.9761904761904763, 0.9761904761904762 , 0.9855275443510737],
                  [0.9747899159663865, 0.9393090569561158, 0.9603174603174602], [0.9626517273576096, 0.9528478057889822 , 0.9831932773109243],
                  [0.953781512605042 , 0.9402427637721755, 0.9897292250233426], [0.9864612511671335, 0.976190476190476  , 0.9640522875816993],
                  [0.7495331465919701, 0.8566760037348273, 0.8398692810457516], [0.8258636788048552, 0.834733893557423  , 0.8120915032679739],
                  [0.7994864612511672, 0.5               , 0.8172268907563026], [0.8081232492997199, 0.8130252100840336 , 0.8209617180205416],
                  [0.8709150326797386, 0.8373015873015873, 0.8370681605975724], [0.8291316526610644, 0.47619047619047616, 0.7362278244631185],
                  [0.8298319327731093, 0.7836134453781513, 0.4757236227824463], [0.7675070028011205, 0.5                , 0.8165266106442578],
                  [0.5               , 0.5894024276377218, 0.5               ], [0.6811391223155929, 0.7394957983193277 , 0.5               ]]
logloss_of_diff_lr = [[0.36389893740728957, 0.3581894420718056 , 0.34242286982081654], [0.3597563502380347 , 0.35374852314660543, 0.36926473816186695],
                      [0.35215721388493065, 0.3599970194574259 , 0.3573829628982003 ], [0.37186002453060074, 0.3650168147648639 , 0.3631919463844666 ],
                      [0.3636412393877985 , 0.38249303309651794, 0.36680583159101593], [0.4002286983552964 , 0.4358719809198711 , 0.4084480519611487 ],
                      [0.3554710246197287 , 0.3557779695740392 , 0.34162770944402243], [0.409657610184645  , 0.382147915322724  , 0.3757167353914878 ],
                      [0.9084755834095378 , 0.9398898966224242 , 1.1281149510567996 ], [0.7153921459316004 , 0.6884846508086607 , 0.6973935542020699 ],
                      [0.825320389888155  , 0.9543245760568466 , 0.783787392074884  ], [1.1158099590635335 , 0.908230258239678  , 0.7161333787773795 ],
                      [0.7333687496196409 , 0.6884614594816046 , 1.0801139357847667 ], [0.7217983709539257 , 0.7459776737887315 , 0.8189323303351931 ],
                      [0.9593478112306022 , 1.1060280365009711 , 1.0571266271424362 ], [0.6904811609963767 , 0.7100202419263232 , 0.7883026863127237 ],
                      [2.6594703020800567 , 0.7809180993533067 , 2.7293530942904636 ], [1.8953579167895798 , 2.321032851126042  , 0.688477070283367  ]]
auc_lobe_lr =  [sum(auc_of_diff_lr[i])/len(auc_of_diff_lr[i]) for i in range(len(auc_of_diff_lr)) if i % 2 == 0]
auc_gyrus_lr = [sum(auc_of_diff_lr[i])/len(auc_of_diff_lr[i]) for i in range(len(auc_of_diff_lr)) if i % 2 == 1]
logloss_lobe_lr = [sum(logloss_of_diff_lr[i])/len(logloss_of_diff_lr[i]) for i in range(len(logloss_of_diff_lr)) if i % 2 == 0]
logloss_gyrus_lr = [sum(logloss_of_diff_lr[i])/len(logloss_of_diff_lr[i]) for i in range(len(logloss_of_diff_lr)) if i % 2 == 1]
assert len(lr_list) == len(auc_lobe_lr) == len(auc_gyrus_lr) == len(logloss_lobe_lr) == len(logloss_gyrus_lr)
# lobe
plt.subplot(2,2,1)
plt.plot(lr_list, auc_lobe_lr, label='AUC', marker='o')
plt.xticks(lr_list,['']*len(lr_list))
plt.title('7 lobe', fontname='Times New Roman', fontsize=20)
for i, value in enumerate(auc_lobe_lr):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.ylabel('AUC', fontname='Times New Roman', fontsize=16)
plt.grid()
plt.subplot(2,2,3)
plt.plot(lr_list, logloss_lobe_lr, label='LogLoss', marker='*', color='orange')
plt.xticks(lr_list, fontname='Times New Roman', fontsize=16)
for i, value in enumerate(logloss_lobe_lr):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.xlabel('learning rate', fontname='Times New Roman', fontsize=16, loc='right')
plt.ylabel('LogLoss', fontname='Times New Roman', fontsize=16)
plt.grid()
# gyrus
plt.subplot(2,2,2)
plt.plot(lr_list, auc_gyrus_lr, label='AUC', marker='o')
plt.xticks(lr_list,['']*len(lr_list))
plt.title('24 gyrus', fontname='Times New Roman', fontsize=20)
for i, value in enumerate(auc_gyrus_lr):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.grid()
plt.subplot(2,2,4)
plt.plot(lr_list, logloss_gyrus_lr, label='LogLoss', marker='*', color='orange')
plt.xticks(lr_list, fontname='Times New Roman', fontsize=16)
for i, value in enumerate(logloss_gyrus_lr):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.xlabel('learning rate', fontname='Times New Roman', fontsize=16, loc='right')
plt.grid()
plt.show()

# learning_rate = 0.0001; epochs vary
# [lobe], [gyrus]
ep_list = ['20', '50', '80', '100', '120', '150', '200']
auc_of_diff_ep = [[0.9234360410830998, 0.9393090569561158, 0.9103641456582634], [0.9575163398692811 , 0.9593837535014006, 0.9598506069094304],
                  [0.9481792717086834, 0.9201680672268907, 0.9551820728291316], [0.9771241830065359 , 0.9715219421101773, 0.9855275443510737],
                  [0.9393090569561158, 0.930905695611578 , 0.9603174603174602], [0.9827264239028944 , 0.9808590102707749, 0.9738562091503267],
                  [0.9547152194211017, 0.9547152194211017, 0.9677871148459383], [0.9813258636788048 , 0.9589169000933706, 0.9803921568627451],
                  [0.9845938375350141, 0.9584500466853407, 0.9621848739495797], [0.9747899159663865 , 0.9794584500466853, 0.9789915966386554],
                  [0.9262371615312791, 0.9505135387488328, 0.9836601307189542], [0.9780578898225956 , 0.9817927170868347, 0.9831932773109243],
                  [0.9523809523809523, 0.9589169000933706, 0.9430438842203548], [0.9794584500466853 , 0.9836601307189542, 0.9822595704948646]]
logloss_of_diff_ep = [[0.44584768111331785, 0.36458482535765613, 0.36012393446080926], [0.35387685325957613, 0.3587985716342382 , 0.35107431642074277],
                      [0.33589669005950806, 0.3416631918313417 , 0.36144201881846916], [0.3497076157981912 , 0.35755573252822537, 0.34758951676467115],
                      [0.3446379144300684 , 0.3619901061890185 , 0.3470498361609182 ], [0.3619481514784985 , 0.3739796902915889 , 0.34873252304724867],
                      [0.3512683513030686 , 0.35007332898690247, 0.36840668397164344], [0.3684867898587656 , 0.36891949423048376, 0.35372899663289686],
                      [0.34137393115006964, 0.36225926055211566, 0.34917416235692644], [0.36702997121922853, 0.37155884078515367, 0.3632852470204971 ],
                      [0.341077993832579  , 0.36092038289545886, 0.3380004156817979 ], [0.39199928545795065, 0.3672952105425706 , 0.3681643144990141 ],
                      [0.36733367203886297, 0.3443161918966849 , 0.36629112191881197], [0.37731049846498776, 0.37625073687830907, 0.3530087373145948]]
auc_lobe_ep =  [sum(auc_of_diff_ep[i])/len(auc_of_diff_ep[i]) for i in range(len(auc_of_diff_ep)) if i % 2 == 0]
auc_gyrus_ep = [sum(auc_of_diff_ep[i])/len(auc_of_diff_ep[i]) for i in range(len(auc_of_diff_ep)) if i % 2 == 1]
logloss_lobe_ep = [sum(logloss_of_diff_ep[i])/len(logloss_of_diff_ep[i]) for i in range(len(logloss_of_diff_ep)) if i % 2 == 0]
logloss_gyrus_ep = [sum(logloss_of_diff_ep[i])/len(logloss_of_diff_ep[i]) for i in range(len(logloss_of_diff_ep)) if i % 2 == 1]
assert len(ep_list) == len(auc_lobe_ep) == len(auc_gyrus_ep) == len(logloss_lobe_ep) == len(logloss_gyrus_ep)
# lobe
plt.subplot(2,2,1)
plt.plot(ep_list, auc_lobe_ep, label='AUC', marker='o')
plt.xticks(ep_list,['']*len(ep_list))
plt.title('7 lobe', fontname='Times New Roman', fontsize=20)
for i, value in enumerate(auc_lobe_ep):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.ylabel('AUC', fontname='Times New Roman', fontsize=16)
plt.grid()
plt.subplot(2,2,3)
plt.plot(ep_list, logloss_lobe_ep, label='LogLoss', marker='*', color='orange')
plt.xticks(ep_list, fontname='Times New Roman', fontsize=16)
for i, value in enumerate(logloss_lobe_ep):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.xlabel('epochs', fontname='Times New Roman', fontsize=16, loc='right')
plt.ylabel('LogLoss', fontname='Times New Roman', fontsize=16)
plt.grid()
# gyrus
plt.subplot(2,2,2)
plt.plot(ep_list, auc_gyrus_ep, label='AUC', marker='o')
plt.xticks(ep_list,['']*len(ep_list))
plt.title('24 gyrus', fontname='Times New Roman', fontsize=20)
for i, value in enumerate(auc_gyrus_ep):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.grid()
plt.subplot(2,2,4)
plt.plot(ep_list, logloss_gyrus_ep, label='LogLoss', marker='*', color='orange')
plt.xticks(ep_list, fontname='Times New Roman', fontsize=16)
for i, value in enumerate(logloss_gyrus_ep):
    plt.text(i, value+0.001, str(round(value,3)), ha='center', va='bottom')
plt.xlabel('epochs', fontname='Times New Roman', fontsize=16, loc='right')
plt.grid()
plt.show()
