# -*- coding: UTF-8 -*-
import re
import matplotlib.pyplot as plt

# lobe_name = ['All', 'Frontal Lobe', 'Temporal Lobe', 'Parietal Lobe', 
#              'Insular Lobe', 'Limbic Lobe', 'Occipital Lobe', 'Subcortical Nuclei']
# lobe_auc = [[0.9705882352941175, 0.9626517273576096, 0.9486461251167133],
#             [0.9295051353874884, 0.9197012138188608, 0.9710550887021476],
#             [0.9117647058823529, 0.8314659197012139, 0.8445378151260503],
#             [0.8748832866479924, 0.8291316526610644, 0.8632119514472456],
#             [0.8949579831932771, 0.8632119514472456, 0.8968253968253967],
#             [0.9126984126984128, 0.9687208216619981, 0.9607843137254901], 
#             [0.8487394957983193, 0.9215686274509803, 0.7675070028011205],
#             [0.8697478991596639, 0.9477124183006536, 0.8902894491129786]]
# lobe_logloss = [[0.3361760814676079 , 0.34869968432606774, 0.35855873308430675],
#                 [0.3129801134555635 , 0.3181792561908074 , 0.2969358745195375],
#                 [0.39561239441507773, 0.5004270381461514 , 0.5117715748266591],
#                 [0.5478438350640282 , 0.49200533095165466, 0.5239538148365399],
#                 [0.4521059868948521 , 0.4551762282587244 , 0.35935263275227064],
#                 [0.3786424321056641 , 0.3080566487336423 , 0.3146869558795491],
#                 [0.5384760521367581 , 0.35587533899984763, 0.5048600722191904],
#                 [0.43384391517388354, 0.3182188387842193 , 0.4829665646186517]]
# lobe_auc = [sum(row)/len(row) for row in lobe_auc]
# lobe_logloss = [sum(row)/len(row) for row in lobe_logloss]
# assert len(lobe_name) == len(lobe_auc) == len(lobe_logloss)

# plt.plot(lobe_name, lobe_auc, label='AUC', marker='o')
# for i, value in enumerate(lobe_auc):
#     plt.text(i, value+0.01, str(round(value,3)), ha='center', va='bottom')
# plt.plot(lobe_name, lobe_logloss, label='LogLoss', marker='*')
# for i, value in enumerate(lobe_logloss):
#     plt.text(i, value+0.01, str(round(value,3)), ha='center', va='bottom')
# plt.title('AUC & LogLoss (7 lobes)', fontname='Times New Roman', fontsize=20)
# plt.legend()
# plt.show()

# All	AUC=0.9705882352941175	LogLoss=0.3361760814676079
res = '''tensor(0.9035)	tensor(0.)
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
numbers = re.findall(r'\d+\.\d*', res)
pred = [float(numbers[i]) for i in range(len(numbers)) if i % 2 == 0]
true = [int(float(numbers[i])) for i in range(len(numbers)) if not i % 2 == 0]
assert len(pred) == len(true)
