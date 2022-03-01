import matplotlib.pyplot as plt 

p1_x = [-0.21412392, -0.015762433, -0.217867797, -0.791634055, -0.954148404, -0.380530783, 0.571634081]
p1_y = [0.49119565, 0.468777474, 0.443375788, 0.758175321, 0.364842911, 0.273814294, -0.032080899]
p2_x = [-1.3457987, -1.235745218, -0.982914495, -0.477240604, -0.640685034, -0.325858797, -0.589265144]
p2_y = [-0.966035, -0.887390127, -0.688952044, -0.25392278, 0.035287042, -0.107229641, -0.056555039]
p3_x = [0.10566617, 0.041052775, 0.113519167, -0.214647122, 0.144898625, -0.013053277, 0.157405465]
p3_y = [0.3526365, 0.442789313, 0.288346561, 0.265619425, -0.084411162, -0.20411088, -0.31204727]
p4_x = [0.7819293, 0.748650813, 0.204286871, 0.339536832, -0.260905726, -0.463877433, -0.625648392]
p4_y = [-0.23768081, -0.255370862, 0.157925975, -0.221139149, 0.613132689, 1.512552275, 2.103614006]

com_x = [-0.223802, -0.220042861, -0.216267826, -0.212492791, -0.208717755, -0.20494272, -0.201167685]
com_y = [-0.15467, -0.091938649, -0.02895452, 0.034029608, 0.097013737, 0.159997866, 0.222981994]

mp1_x = [-0.21412392, -0.134910971, -0.300173134, -0.929729879, -0.948537469, -0.439163476, 0.741447568]
mp1_y = [0.49119565, 0.531131566, 0.439975649, 0.631887436, 0.367041826, 0.218266338, -0.293207079]
mp2_x = [-1.3457987, -1.239014268, -1.085573554, -0.500737131, -0.458262712, -0.400236726, -0.518132865]
mp2_y = [-0.966035, -0.911953211, -0.717689991, -0.24884215, 0.086513564, -0.090855449, -0.089281]
mp3_x = [0.10566617, 0.059930045, 0.088881776, -0.278298259, 0.096123993, -0.042111371, 0.130468428]
mp3_y = [0.3526365, 0.359143466, 0.332171112, 0.189035147, 0.036286511, -0.154774159, -0.2745772]
mp4_x = [0.7819293, 0.712825358, 0.230800956, 0.367553532, -0.253054649, -0.338851869, -0.696606874]
mp4_y = [-0.23768081, -0.302985787, 0.072461098, -0.185990512, 0.565921485, 1.570806265, 2.205282927]

mcom_x = [-0.223802, -0.222890928, -0.257293165, -0.249513701, -0.174470022, -0.220359921, -0.199401915]
mcom_y = [-0.15467, -0.145751387, -0.032886054, 0.001338521, 0.161255211, 0.197241902, 0.24043411]

fig, ax = plt.subplots()
"""
plt.plot(p1_x, p1_y, label='Body 1')
plt.plot(p2_x, p2_y, label='Body 2')
plt.plot(p3_x, p3_y, label='Body 3')
plt.plot(p4_x, p4_y, label='Body 4')
plt.plot(com_x, com_y, label="System Center of Mass")
"""
plt.plot(mp1_x, mp1_y, label='Body 1')
plt.plot(mp2_x, mp2_y, label='Body 2')
plt.plot(mp3_x, mp3_y, label='Body 3')
plt.plot(mp4_x, mp4_y, label='Body 4')
plt.plot(mcom_x, mcom_y, label='System Center of Mass')

plt.legend()
plt.show()


"""
p1_x = []
p1_y = []
p2_x = []
p2_y = []
p3_x = []
p3_y = []
p4_x = []
p4_y = []
"""