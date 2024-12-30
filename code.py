import numpy as np
import scipy.io

def calculate_true_cleans(cleans):
    true_cleans = np.zeros(365)
    ranges = [39, 79, 106, 121, 150, 162, 178, 193, 206, 365]
    for i in range(365):
        for r in ranges:
            if i < r:
                true_cleans[i] = r - i


    return true_cleans

file_path = 'Cleansing.mat'
mat_data = scipy.io.loadmat(file_path)
cleans = mat_data['Cleansing']
true_cleans = calculate_true_cleans(cleans)
true_cleans = np.repeat(true_cleans, 24)



file_path = 'day.mat'
days = np.zeros(365)
for i in range(365):
    days[i] = i
days = np.repeat(days, 24)

file_path = 'Radiation1400New.mat'
mat_data = scipy.io.loadmat(file_path)
Radiation1400New = np.array(mat_data['Radiation1400New'])



file_path = 'Rain.mat'
mat_data = scipy.io.loadmat(file_path)
Rain = np.array(mat_data['Rain1400'])
Rain = np.repeat(Rain, 24)


file_path = 'Temperature.mat'
mat_data = scipy.io.loadmat(file_path)
Temperature = np.array(mat_data['Temperature1400'])
Temperature = np.repeat(Temperature, 3)

file_path = 'Power.mat'
mat_data = scipy.io.loadmat(file_path)
Power = np.array(mat_data['Power1400'])
Power = Power.reshape((24*365,1))

true_cleans = true_cleans.reshape((24*365,1))
days = days.reshape((24*365,1))
Radiation1400New = Radiation1400New.reshape((24*365,1))
Rain = Rain.reshape((24*365,1))
Temperature = Temperature.reshape((24*365,1))

input_data = np.concatenate((true_cleans,days,Radiation1400New,Rain,Temperature),axis=0)
output_data = Power