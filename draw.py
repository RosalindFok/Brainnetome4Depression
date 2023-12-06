# -*- coding: UTF-8 -*-
import re
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
gyrus_auc = [[0.9719887955182072, 0.9869281045751633, 0.9822595704948646],
             [0.9313725490196078, 0.9206349206349206, 0.8851540616246498],
             [0.8356676003734828, 0.9108309990662931, 0.8674136321195145],
             [0.8940242763772176, 0.7791783380018673, 0.8356676003734826],
             [0.9262371615312791, 0.927170868347339 , 0.919234360410831 ],
             [0.8160597572362278, 0.8076563958916899, 0.7815126050420169],
             [0.896358543417367 , 0.8818860877684408, 0.8879551820728292],
             [0.9477124183006536, 0.9183006535947712, 0.957516339869281 ],
             [0.8594771241830065, 0.8258636788048552, 0.9056956115779645],
             [0.9663865546218487, 0.9542483660130718, 0.9733893557422968],
             [0.8893557422969187, 0.8123249299719888, 0.8940242763772175],
             [0.8814192343604108, 0.9112978524743232, 0.8986928104575163],
             [0.945845004668534 , 0.9631185807656396, 0.9827264239028944],
             [0.8655462184873949, 0.8328664799253034, 0.8053221288515405],
             [0.8272642390289449, 0.8739495798319328, 0.7880485527544352],
             [0.9243697478991597, 0.907563025210084 , 0.8935574229691877],
             [0.699813258636788 , 0.6568627450980392, 0.6918767507002801],
             [0.9234360410830997, 0.942110177404295 , 0.865546218487395 ],
             [0.9547152194211017, 0.9500466853408029, 0.8954248366013072],
             [0.8099906629318394, 0.746031746031746 , 0.7899159663865546],
             [0.9014939309056955, 0.8281979458450046, 0.8473389355742297],
             [0.9640522875816993, 0.9864612511671336, 0.9813258636788049],
             [0.9187675070028012, 0.8557422969187676, 0.9584500466853407],
             [0.903828197945845 , 0.9411764705882353, 0.9533146591970121],
             [0.9108309990662933, 0.9075630252100839, 0.8772175536881419]]
gyrus_logloss = [[0.3774568150507361 , 0.364157707112194  , 0.3723116611934127 ],
                 [0.35641000370436227, 0.4387082121543099 , 0.4804407880045864 ],
                 [0.409206209779479  , 0.35103885495338466, 0.37872649610253006],
                 [0.3584008252091755 , 0.36476627337494427, 0.37162705644995914],
                 [0.39994568925406604, 0.4064223906594925 , 0.408520794768198  ],
                 [0.3709179724282271 , 0.3939653873730284 , 0.37208563503834163],
                 [0.5733255667532833 , 0.6333083538001606 , 0.5665857667719205 ],
                 [0.27907755687412555, 0.40647827490230826, 0.25793459571270516],
                 [0.376753921521312  , 0.4385845348296728 , 0.39721109670923954],
                 [0.32138021727191757, 0.39352877031072236, 0.23381144263576623],
                 [0.38632832079410784, 0.41784797276125724, 0.4108179746891131 ],
                 [0.4452729877117041 , 0.3849089769696607 , 0.44990139522614153],
                 [0.37068465744240137, 0.32745084416815795, 0.25871289208638387],
                 [0.43247118947325647, 0.47532718542942426, 0.5312557474495787 ],
                 [0.46440942277137043, 0.4301231244324399 , 0.4561651684115128 ],
                 [0.4604867791661369 , 0.4802081290096029 , 0.48592302320701863],
                 [0.5726964390556203 , 0.5963472852342013 , 0.6537600203944443 ],
                 [0.3776411889909457 , 0.380201783668372  , 0.4154002054953348 ],
                 [0.4194559007895425 , 0.42221918831614125, 0.5549616972390864 ],
                 [0.5753047883581125 , 0.5011434518322165 , 0.4736773478042191 ],
                 [0.3601178345332771 , 0.3818329795897665 , 0.3816303475202984 ],
                 [0.2949693617759361 , 0.2473291464539728 , 0.25645020680398334],
                 [0.3651515853068269 , 0.3763668066731855 , 0.34943384011011536],
                 [0.3689283083630569 , 0.34176944614580346, 0.37119242383323475],
                 [0.4486134269595081 , 0.46181930495037327, 0.4562234368859291 ]]
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