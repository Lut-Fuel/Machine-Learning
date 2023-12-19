import pandas as pd
import numpy as np

#Read the data file
df = pd.read_csv('Dataset/Raw Data/Mobil Indonesia.csv')

#Dropping Null and unused data
df.dropna(inplace=True)
df = df[~df['Model'].str.contains('Hybrid')]

#Processing engine type data
engine_to_remove = ['Battery Electric', 'Electric', 'Hybrid', 'Fuel Cell']
df = df[~df['Engine Type'].isin(engine_to_remove)]
df['Engine Type'] = df['Engine Type'].replace({
    'Turbocharged Gasoline Direct Injection' : 0,
    'Gasoline Direct Injection' : 0,
    'Petrol' : 0,
    'Dual Variable Valve Timing-intelligent' : 0,
    'Multi Point Injection' : 0,
    'Turbocharged Diesel' : 1,
    'Diesel/Petrol/Plug-in Hybrid' : 1,
    'Diesel/Petrol/Mild Hybrid' : 1,
    'Diesel' : 1,
    'Direct Injection' : 0
})

#Processing transmission type data
df['Transmission'] = df['Transmission'].replace({
    '7-speed dual clutch': 1,
    '8-speed dual clutch': 1,
    '8-speed automatic': 0,
    '6-speed automatic': 0,
    '9-speed automatic': 0,
    'Automatic': 0,
    '4-speed automatic': 0,
    'Continuously Variable Transmission': 0,
    '10-speed automatic': 0,
    '5-speed manual': 1,
    '6-speed manual': 1,
    '6-speed dual clutch': 1,
    '8-speed Automatic': 0,
    '1-speed direct drive': 0,
    'Manual': 1
})

#Processing car type data
df['Type of Car'] = df['Type of Car'].replace({
    'Sedan': 0,
    'Saloon': 0,
    'Coupe': 0,
    'Convertible': 0,
    'Coupe/Convertible': 0,
    'SUV': 1,
    'Hatchback': 1,
    'MPV': 2,
    'Minivan': 2,
    'Van': 2,
    'Estate': 2,
    'Pickup': 3
})
df['Type of Car'] = df['Type of Car'].astype(int)

#Processing fuel tank capacity data
df['Fuel Tank Capacity'] = df['Fuel Tank Capacity'].str.replace(' L', '')
df['Fuel Tank Capacity'] = df['Fuel Tank Capacity'].replace({
    '56-69': np.mean([56, 69]),
    '60-82': np.mean([60, 82]),
    '63-70': np.mean([63, 70]),
    '56-63': np.mean([56, 63])
})
df['Fuel Tank Capacity'] = df['Fuel Tank Capacity'].astype(int)

#Processing acceleration data
df['Acceleration 0 to 100 Km'] = df['Acceleration 0 to 100 Km'].str.replace(' s', '')

#Processing fuel grade data
df['Fuel Grade (in Octane)'] = df['Fuel Grade (in Octane)'].replace({
    '95': 3,
    '98': 3,
    '91': 1,
    '87': 0,
    '93': 2,
    '95.0': 3,
    '98.0': 3,
    '95-98': 3,
    '87.0': 0,
    '93.0': 2,
    '91.0': 1
})
df['Fuel Grade (in Octane)'] = df['Fuel Grade (in Octane)'].astype(int)

#Processing engine hp data
def kW_to_hp(power):
    horse_power = 1.3596216173*power
    horse_power = np.floor(horse_power)
    return horse_power

df['Engine Horse Power'] = df['Engine Horse Power'].replace({
    '140 kW': kW_to_hp(140),
    '74 kW': kW_to_hp(74),
    '126 kW': kW_to_hp(126),
    '77 kW': kW_to_hp(77),
    '110 kW': kW_to_hp(110),
    '81 kW': kW_to_hp(81),
    '96 kW': kW_to_hp(96),
    '133 kW': kW_to_hp(133)
})

df['Engine Horse Power'] = df['Engine Horse Power'].replace({
    '163-300': np.mean([163, 300]),
    '300-575': np.mean([300, 575]),
    '163-550': np.mean([163, 550])
})

df['Engine Horse Power'] = df['Engine Horse Power'].astype(float)

#Processing cylinder data
df['Number of Cylinders'] = df['Number of Cylinders'].replace({
    '4.0': 4,
    '6.0': 6,
    '8.0': 8,
    '10.0': 10,
    '3.0': 3,
    '4-8': 4
})
df['Number of Cylinders'] = df['Number of Cylinders'].astype(int)

#Processing engine hp rpm data
df['Engine Horse Power RPM'] = df['Engine Horse Power RPM'].replace({
    '3500-6500': np.mean([3500, 6500]),
    '5500-6500': np.mean([5500-6500]),
    '3500-5500': np.mean([3500, 5500])
})
df['Engine Horse Power RPM'] = df['Engine Horse Power RPM'].astype(float)
df['Engine Horse Power RPM'] = df['Engine Horse Power RPM'].astype(int)

#Processing max speed data
df['Max Speed Km per Hour'] = df['Max Speed Km per Hour'].replace({
    '180-243 km/h': np.mean([180, 243]),
    '180-286 km/h': np.mean([180, 286]),
    '250-300 km/h': np.mean([250, 300]),
    '230-250 km/h': np.mean([230, 250]),
})
df['Max Speed Km per Hour'] = df['Max Speed Km per Hour'].astype(float)
df['Max Speed Km per Hour'] = df['Max Speed Km per Hour'].astype(int)

#Renaming columns by adding '_'
df.columns = df.columns.str.replace(' ', '_')
df.rename(columns={'Fuel_Grade_(in_Octane)': 'Fuel_Grade'}, inplace=True)
df.rename(columns={'Mixed_Fuel_Consumption_per_Km': 'Mixed_Fuel_Consumption_per_100_km_l'}, inplace=True)

#Processing the mixed fuel consumption, which is the target
df['Mixed_Fuel_Consumption_per_100_km_l'] = df['Mixed_Fuel_Consumption_per_100_km_l'].str.replace('L/', '')
df['Mixed_Fuel_Consumption_per_100_km_l'] = df['Mixed_Fuel_Consumption_per_100_km_l'].str.replace('km', '')
df[['Consumption_L_per_100_km', 'Distance_km']] = df['Mixed_Fuel_Consumption_per_100_km_l'].str.split(expand=True)
df['Consumption_L_per_100_km'] = pd.to_numeric(df['Consumption_L_per_100_km'], errors='coerce')
df['Distance_km'] = pd.to_numeric(df['Distance_km'], errors='coerce')
df['Consumption_km_per_L'] = df['Distance_km'] / df['Consumption_L_per_100_km']
df = df.drop(['Consumption_L_per_100_km', 'Distance_km', 'Mixed_Fuel_Consumption_per_100_km_l'], axis=1)
df.rename(columns={'Consumption_km_per_L': 'Mixed_Fuel_Consumption_per_100_km_l'}, inplace=True)
df['Mixed_Fuel_Consumption_per_100_km_l'] = df['Mixed_Fuel_Consumption_per_100_km_l'].round(1)


print(df.info())
print('')
print(df.head())

#Saving the cleaned data
df.to_csv('Dataset/indo_car_train_data.csv', index=False)

#Dropping the mixed fuel consumption data for the database
df = df.drop('Mixed_Fuel_Consumption_per_100_km_l', axis=1)

#Combining Maker and Model data into new column 'Car Name'
df['Car Name'] = pd.concat([df['Maker'], df['Model']], axis=1).agg(' '.join, axis=1)

#Saving the database
df.to_csv('Dataset/database_indo.csv', index=False)